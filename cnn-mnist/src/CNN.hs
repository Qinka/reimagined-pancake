{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE OverloadedLists   #-}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE DataKinds         #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module CNN where

import Control.Monad
import Control.Monad.IO.Class
import Data.Int
import Data.Word
import Data.List
import           Lens.Family2
import qualified TensorFlow.Output      as TF (OpDef (..), opAttr)
import qualified TensorFlow.Types       as TF
import qualified TensorFlow.BuildOp     as TF (OpParams (..))
import qualified Data.Text.IO as T
import qualified Data.Vector as V
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF hiding ( initializedVariable
                                             , zeroInitializedVariable
                                             )
import qualified TensorFlow.Variable as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.GenOps.Core as TF (conv2D',maxPool')
import qualified Data.ByteString as B

import Parser.MNIST


numPixels, numLabels :: Int64
numPixels = 28 * 28
numLabels = 10

randomParam :: Int64 -> TF.Shape -> TF.Build (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) =
    (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

data Model = Model
             { train :: TF.TensorData Image
                     -> TF.TensorData Label
                     -> TF.Session ()
             , infer :: TF.TensorData Image
                     -> TF.Session (V.Vector Label)
             , errRt :: TF.TensorData Image
                     -> TF.TensorData Label
                     -> TF.Session Float
             }

 -- conv2D
conv2D :: TF.OneOf '[Word16,Float] t
       => TF.Tensor v'1   t -- ^ input
       -> TF.Tensor v'2   t -- ^ filter
       -> TF.Tensor TF.Build t -- ^ output
conv2D = TF.conv2D'
  ( (TF.opAttr "strides"            .~ ([1,1,1,1] :: [Int64]))
    . (TF.opAttr "use_cudnn_on_gpu" .~ True)
    . (TF.opAttr "padding"          .~ ("SAME" :: B.ByteString))
    . (TF.opAttr "data_format"      .~ ("NHWC" :: B.ByteString))
  )

-- max pool
maxPool :: TF.OneOf '[Word16,Float] t
        => (Int64,Int64) -- ^ ksize
        -> (Int64,Int64) -- ^ stride
        -> TF.Tensor v'1   t -- ^ input
        -> TF.Tensor TF.Build t -- ^ output
maxPool (k1,k2) (s1,s2) = TF.maxPool'
  ( (TF.opAttr "ksize"       .~ ([1,k1,k2,1] :: [Int64]))
    . (TF.opAttr "strides"     .~ ([1,s1,s2,1] :: [Int64]))
    . (TF.opAttr "padding"     .~ ("SAME" :: B.ByteString))
    . (TF.opAttr "data_format" .~ ("NHWC" :: B.ByteString))
  )

createModel :: TF.Build Model
createModel = do
  let value x y = TF.initializedVariable =<< randomParam x y
  -- traning data
  let batchSize = -1
  images <- TF.placeholder [batchSize, numPixels]
  -- layer 1 (conv)
  -- params
  w1 <- value 3 [3,3,1,16]
  b1 <- TF.zeroInitializedVariable [16]
  -- conv
  let c1 = images `conv2D` TF.readValue w1 `TF.add` TF.readValue b1
  -- ReLU
  let r1 = TF.relu c1
  -- max pool
  let m1 = maxPool (2,2) (2,2) r1
  -- layer 2 (conv)
  -- params
  w2 <- value 3 [3,3,16,32]
  b2 <- TF.zeroInitializedVariable [32]
  -- conv
  let c2 = m1 `conv2D` TF.readValue w2 `TF.add` TF.readValue b2
  -- ReLU
  let r2 = TF.relu c2
  -- max pool
  let m2 = maxPool (2,2) (2,2) r2
  -- trans
  let m2' = TF.reshape m2 $ TF.constant [2] [-1,1568 :: Int32]
  -- layer 3 (full connection)
  -- params
  w3 <- value 1568 [1568,128]
  b3 <- TF.zeroInitializedVariable [128]
  -- fc
  let f3 = m2' `TF.matMul` TF.readValue w3 `TF.add` TF.readValue b3
  -- ReLU
  let r3 = TF.relu f3
  -- layer 4 (full connection)
  -- params
  w4 <- value 128 [128,10]
  b4 <- TF.zeroInitializedVariable [10]
  -- fc
  let f4 = r3 `TF.matMul` TF.readValue w4 `TF.add` TF.readValue b4
  -- ReLU
  let r4 = TF.relu f4
  -- predict
  predict <- TF.render $ TF.cast $
             (TF.argMax (TF.softmax r4) (TF.scalar (1 :: Label)) ::TF.Tensor TF.Build Int64 ) 
  -- label
  labels <- TF.placeholder [batchSize]
  let labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0
      loss = TF.reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits r4 labelVecs
      params = [w1,b1,w2,b2,w3,b3,w4,b4]
  trainStep <- TF.minimizeWith TF.adam loss params
  let correctPrediections = predict `TF.equal` labels
  errorRate <- TF.render $ 1 - TF.reduceMean (TF.cast correctPrediections)
  -- return model
  return Model
    { train = \imF lF -> TF.runWithFeeds_
                         [ TF.feed images imF
                         , TF.feed labels lF
                         ] trainStep
    , infer = \imF -> TF.runWithFeeds
                      [ TF.feed images imF
                      ] predict
    , errRt = \imF lF -> TF.unScalar <$> TF.runWithFeeds
                         [ TF.feed images imF
                         , TF.feed labels lF
                         ] errorRate
    }



mainSess :: [MNIST] -- ^ train image
         -> [Label] -- ^ train label
         -> [MNIST] -- ^ test  image
         -> [Label] -- ^ test label
         -> FilePath -- store file
         -> Bool -- ^ whether restore
         -> Int -- ^ times
         -> TF.Session ()
mainSess trI trL teI teL fp isRe times = do
  model <- TF.build createModel
  let encodeImageBatch xs =
        TF.encodeTensorData [genericLength xs, 28, 28,1] $ mconcat xs
      encodeLabelBatch xs =
        TF.encodeTensorData [genericLength xs] $  V.fromList xs
      batchSize = 100
      selectBatch i xs = take batchSize $ drop (i * batchSize) (cycle xs)
      t = times * 100
  forM_ ([0..t] :: [Int]) $ \i -> do
    let images = encodeImageBatch (selectBatch i trI)
        labels = encodeLabelBatch (selectBatch i trL)
    train model images labels
    when (i `mod` 100 == 0) $ do
      err <- errRt model images labels
      liftIO $ putStrLn $ "training error(" ++ show i ++ ")" ++ show (err * 100)
  liftIO $ putStrLn ""
  testErr <- errRt model
             (encodeImageBatch teI)
             (encodeLabelBatch teL)
  liftIO $ putStrLn $ "test error " ++ show (testErr * 100)


runMain a b c d e f g = TF.runSession $ mainSess a b c d e f g
