
\section{Handwritten Numeral Recognition with MNIST}
\label{sec:hnr}

\begin{code}
{-# LANGUAGE FlexibleContexts  #-}
{-# LANGUAGE OverloadedLists   #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE OverloadedStrings #-}

module AI.MNIST
  ( train_mnist
  , TrainModel(..)
  , InferModel(..)
  , ModelParams(..)
  , createTrainModel
  , createInferModel
  ) where

import AI.MNIST.Parse
import Control.Monad (forM_, when)
import Control.Monad.IO.Class (liftIO)
import Data.Int (Int32, Int64)
import Data.Word (Word8)
import Data.List (genericLength)
import qualified Data.Text.IO as T
import qualified Data.Vector as V
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable, zeroInitializedVariable)
import qualified TensorFlow.Variable as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.GenOps.Core as TF (conv2D',maxPool')
import qualified TensorFlow.BuildOp as TF (OpParams(..))
import qualified TensorFlow.Output as TF (OpDef(..),opAttr)
import qualified Data.Map as Map
import Lens.Family2
import qualified Data.ByteString as B
\end{code}

In this section, the main topic is about ``handwritten numeral recognition'' with MNIST%
\footnote{Short for Modified National Institue of Standards and Technology database.}


The ``tasks'', or say layouts of recognition with convolution neural network are included the following items:
\begin{itemize}
\item convolution
\item hidden
\item hidden
\item output
\end{itemize}


\begin{code}
-- | number of pixels
numPixels :: Int64
numPixels = 28
-- | number of label
numLabels :: Int64
numLabels = 10

-- | create random params
randomParam :: Int64 -> TF.Shape -> TF.Build (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) = (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where stddev = TF.scalar (1 / sqrt (fromIntegral width))


-- | node sizes
nodeSizes :: [Int64]
nodeSizes = [5,1440,720,360,10]

featureSize :: Int64
featureSize = 10

-- | type of label
type Label = Int32

-- | Model params
data ModelParams = ModelParams
                   (V.Vector Float) -- ^ w1
                   (V.Vector Float) -- ^ b1
                   (V.Vector Float) -- ^ w2
                   (V.Vector Float) -- ^ b2
                   (V.Vector Float) -- ^ w3
                   (V.Vector Float) -- ^ b3
                   (V.Vector Float) -- ^ w4
                   (V.Vector Float) -- ^ b4
                   deriving Show
                   

-- | define model of training
data TrainModel = TrainModel { train :: TF.TensorData Float -- ^ images
                                     -> TF.TensorData Label -- ^ Label
                                     -> TF.Session ()
                             , infer :: TF.TensorData Float -- ^ images
                                     -> TF.Session (V.Vector Label) -- ^ labels
                             , errRt :: TF.TensorData Float -- ^ images
                                     -> TF.TensorData Label -- ^ labels
                                     -> TF.Session Float
                             , param :: TF.Session ModelParams
                             }
-- | define the model of infering
data InferModel = InferModel { doInfer :: TF.TensorData Float -- ^ images
                                       -> TF.Session (V.Vector Label) -- ^ labels
                             }
-- | create common model
commonModel :: TF.MonadBuild m
            => Int64 -- ^ batch size
            -> m ( TF.Variable Float, TF.Variable Float) -- ^ matrix and bias for layer 1 (cnn)
            -> m ( TF.Variable Float, TF.Variable Float) -- ^ weight and bias for layer 2 
            -> m ( TF.Variable Float, TF.Variable Float) -- ^ weight and bias for layer 3
            -> m ( TF.Variable Float, TF.Variable Float) -- ^ weight and bias for layer 4 (logit)
            -> m ( TF.Tensor TF.Value Float -- ^ images
                 , TF.Tensor TF.Value Label -- ^ predict
                 , ( TF.Variable Float, TF.Variable Float) -- ^ matrix and bias
                 , ( TF.Variable Float, TF.Variable Float) -- ^ weight and bias for hidden layer 1
                 , ( TF.Variable Float, TF.Variable Float) -- ^ weight and bias for hidden layer 2
                 , ( TF.Variable Float, TF.Variable Float) -- ^ weight and bias for logit layer
                 , TF.Tensor TF.Build Float -- ^ logit
                 )
commonModel batchSize l1 l2 l3 l4 = do
  images <- TF.placeholder [batchSize,numPixels,numPixels,1]
  let conv2D = TF.conv2D' ( (TF.opAttr "strides" .~ ([1,1,1,1] :: [Int64]))
                          . (TF.opAttr "use_cudnn_on_gpu" .~ True)
                          . (TF.opAttr "padding" .~ ("VALID" :: B.ByteString))
                          . (TF.opAttr "data_format" .~ ("NHWC" :: B.ByteString))
                          ) 
      maxPool = TF.maxPool' ( (TF.opAttr "ksize"   .~ ([1,2,2,1] :: [Int64]))
                            . (TF.opAttr "strides" .~ ([1,2,2,1] :: [Int64]))
                            . (TF.opAttr "padding" .~ ("VALID" :: B.ByteString))
                            . (TF.opAttr "data_format" .~ ("NHWC" :: B.ByteString))
                            )
  -- layer 1 (cnn)
  (w1,b1) <- l1
  let l1Exp' = maxPool $ images `conv2D` TF.readValue w1 `TF.add` TF.readValue b1
      l1Exp  = TF.reshape l1Exp' (TF.constant [2] [-1,1440 :: Int32])
  (w2,b2) <- l2
  let l2Exp = l1Exp  `TF.matMul` TF.readValue w2 `TF.add` TF.readValue b2
  (w3,b3) <- l3
  let l3Exp = l2Exp  `TF.matMul` TF.readValue w3 `TF.add` TF.readValue b3
  (w4,b4) <- l4
  let l4Exp = l3Exp  `TF.matMul` TF.readValue w4 `TF.add` TF.readValue b4
  predict <- TF.render $ TF.cast $ TF.argMax (TF.softmax l4Exp) (TF.scalar (1 :: Int64))
  return (images,predict,(w1,b1),(w2,b2),(w3,b3),(w4,b4),l4Exp)

createInferModel :: [[Float]] -> TF.Build InferModel
createInferModel (w1:b1:w2:b2:w3:b3:w4:b4:_) = do
  let batchSize = -1
  (images,predict,(w1,b1),(w2,b2),(w3,b3),(w4,b4),_) <-
    commonModel batchSize
    ((,) <$> (TF.initializedVariable $ TF.constant [nodeSizes !! 0,nodeSizes !! 0, 1, featureSize] w1)
         <*> (TF.initializedVariable $ TF.constant [                  featureSize] b1))
    ((,) <$> (TF.initializedVariable $ TF.constant [nodeSizes !! 1,nodeSizes !! 2] w2)
         <*> (TF.initializedVariable $ TF.constant [               nodeSizes !! 2] b2))
    ((,) <$> (TF.initializedVariable $ TF.constant [nodeSizes !! 2,nodeSizes !! 3] w3)
         <*> (TF.initializedVariable $ TF.constant [               nodeSizes !! 3] b3))
    ((,) <$> (TF.initializedVariable $ TF.constant [nodeSizes !! 3,nodeSizes !! 4] w4)
         <*> (TF.initializedVariable $ TF.constant [               nodeSizes !! 4] b4))
  return InferModel { doInfer = \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict
                    }
createTrainModel :: TF.Build TrainModel
createTrainModel = do
  let batchSize = -1
  (images,predict,(w1,b1),(w2,b2),(w3,b3),(w4,b4),logits) <-
    commonModel batchSize
    ((,) <$> (TF.initializedVariable =<< randomParam (nodeSizes !! 0)  [nodeSizes !! 0, nodeSizes !! 0, 1, featureSize])
         <*> (TF.zeroInitializedVariable [featureSize]))
    ((,) <$> (TF.initializedVariable =<< randomParam (nodeSizes !! 1)  [nodeSizes !! 1, nodeSizes !! 2])
         <*> (TF.zeroInitializedVariable [nodeSizes !! 2]))
    ((,) <$> (TF.initializedVariable =<< randomParam (nodeSizes !! 2)  [nodeSizes !! 2, nodeSizes !! 3])
         <*> (TF.zeroInitializedVariable [nodeSizes !! 3]))
    ((,) <$> (TF.initializedVariable =<< randomParam (nodeSizes !! 3)  [nodeSizes !! 3, nodeSizes !! 4])
         <*> (TF.zeroInitializedVariable [nodeSizes !! 4]))
  labels <- TF.placeholder [batchSize]
  let labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0
      loss      = TF.reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
      params    = [w1,b1,w2,b2,w3,b3,w4,b4]
  trainStep <- TF.minimizeWith TF.adam loss params
  let  correctPredictions = TF.equal predict labels
  errorRateTensor <- TF.render $ 1 - TF.reduceMean (TF.cast correctPredictions)
  return TrainModel { train = \imFeed lFeed -> TF.runWithFeeds_ [ TF.feed images imFeed
                                                                , TF.feed labels lFeed
                                                                ] trainStep
                    , infer = \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict
                    , errRt = \imFeed lFeed -> TF.unScalar <$> TF.runWithFeeds [ TF.feed images imFeed
                                                               , TF.feed labels lFeed
                                                               ] errorRateTensor
                    , param = let trans :: TF.Variable Float -> TF.Session (V.Vector Float)
                                  trans x = TF.run =<< TF.render (TF.cast $ TF.readValue x)
                              in ModelParams
                                 <$> trans w1
                                 <*> trans b1
                                 <*> trans w2
                                 <*> trans b2
                                 <*> trans w3
                                 <*> trans b3
                                 <*> trans w4
                                 <*> trans b4
                    }
                                               
train_mnist :: Int -> [MNIST] -> [Word8] -> [MNIST] -> [Word8] -> IO ModelParams
train_mnist times sample label ts tl = TF.runSession $ do
  model <- TF.build createTrainModel
  let encodeImageBatch xs = TF.encodeTensorData [genericLength xs, numPixels, numPixels,1]
                            (fromIntegral <$>    mconcat xs)
      encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (fromIntegral <$> V.fromList xs)
      batchSize           = 100
      selectBatch    i xs = take batchSize $ drop (i * batchSize) (cycle xs)
  forM_ ([0..times] :: [Int]) $ \i -> do
    let images = encodeImageBatch (selectBatch i sample)
        labels = encodeLabelBatch (selectBatch i  label)
    train model images labels
    when (i `mod` 100 == 0) $ do
      err <- errRt model images labels
      liftIO $ putStrLn $ show (i `div` 100)  ++ " training error: " ++ show (err * 100) ++ "%\n"
  let images = encodeImageBatch ts
      labels = encodeLabelBatch tl
  errTest <- errRt model images labels
  liftIO $ putStrLn $ "training error(for training set): " ++ show (errTest * 100) ++ "%\n"
  param model   
\end{code}

The following is the example command for training with mnist
\begin{spec}
let path'prefix = ""
s  <- readMNISTSamples $ path'prefix ++ "train-images-idx3-ubyte.gz"
l  <- readMNISTLabels  $ path'prefix ++ "train-labels-idx1-ubyte.gz"
ts <- readMNISTSamples $ path'prefix ++  "t10k-images-idx3-ubyte.gz"
tl <- readMNISTLabels  $ path'prefix ++  "t10k-labels-idx1-ubyte.gz"
train_mnist 1500 s l ts tl
\end{spec}
