
\section{Handwritten Numeral Recognition with MNIST}
\label{sec:hnr}

\begin{code}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists  #-}
{-# LANGUAGE TemplateHaskell  #-}

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
import qualified TensorFlow.GenOps.Core as TF (conv2D,maxPool)
\end{code}

In this section, the main topic is about ``handwritten numeral recognition'' with MNIST%
\footnote{Short for Modified National Institue of Standards and Technology database.}


The ``tasks'', or say layouts of recognition with convolution nerual network are included the following items:
\begin{itemize}
\item convolution
\item hidden
\item hidden
\item output
\end{itemize}


\begin{code}
-- | number of pixels
numPixels :: Int64
numPixels = 28 * 28
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
  images <- TF.placeholder [batchSize,numPixels]
  -- layer 1 (cnn)
  (w1,b1) <- l1
  let l1Exp = TF.maxPool $ images `TF.conv2D` TF.readValue w1 `TF.add` TF.readValue b1
  (w2,b2) <- l2
  let l2Exp = l1Exp  `TF.matMul` TF.readValue w2 `TF.add` TF.readValue b2
  (w3,b3) <- l3
  let l3Exp = l2Exp  `TF.matMul` TF.readValue w3 `TF.add` TF.readValue b3
  (w4,b4) <- l4
  let l4Exp = l4Exp  `TF.matMul` TF.readValue w4 `TF.add` TF.readValue b4
  predict <- TF.render $ TF.cast $ TF.argMax (TF.softmax l4Exp) (TF.scalar (1 :: Int64))
  return (images,predict,(w1,b1),(w2,b2),(w3,b3),(w4,b4),l4Exp)

createInferModel :: [[Float]] -> TF.Build InferModel
createInferModel (w1:b1:w2:b2:w3:b3:w4:b4:_) = do
  let batchSize = -1
  (images,predict,(w1,b1),(w2,b2),(w3,b3),(w4,b4),_) <-
    commonModel batchSize
    ((,) <$> (TF.initializedVariable $ TF.constant [nodeSizes !! 0,nodeSizes !! 0] w1)
         <*> (TF.initializedVariable $ TF.constant [               nodeSizes !! 1] b1))
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
    ((,) <$> (TF.initializedVariable =<< randomParam (nodeSizes !! 0)  [nodeSizes !! 0, nodeSizes !! 0])
         <*> (TF.zeroInitializedVariable [nodeSizes !! 1]))
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
                                               
train_mnist :: [MNIST] -> [Word8] -> IO ModelParams
train_mnist sample label = TF.runSession $ do
  model <- TF.build createTrainModel
  let encodeImageBatch xs = TF.encodeTensorData [genericLength xs, numPixels] (fromIntegral <$>    mconcat xs)
      encodeLabelBatch xs = TF.encodeTensorData [genericLength xs, numPixels] (fromIntegral <$> V.fromList xs)
      batchSize           = 100
      selectBatch    i xs = take batchSize $ drop (i * batchSize) (cycle xs)
  forM_ ([0..1000] :: [Int]) $ \i -> do
    let images = encodeImageBatch (selectBatch i sample)
        labels = encodeLabelBatch (selectBatch i  label)
    train model images labels
    when (i `mod` 100 == 0) $ do
      err <- errRt model images labels
      liftIO $ putStrLn $ "training error: " ++ show (err * 100) ++ "%\n"
  param model
    
  

    
\end{code}
