
\section{Neural Network for Human Face Judger}
\label{sec:nnfhfj}

\begin{code}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists  #-}
{-# LANGUAGE TemplateHaskell  #-}
module AI.NN
       ( TrainModel(..)
       , PredictModel(..)
       , PredictModelParam(..)
       , createTrainModel
       , createPredictModel
       , runPredict
       , training
       ) where


import           Control.Monad          (zipWithM, when, forM_)
import           Control.Monad.IO.Class (liftIO)
import           Data.Aeson.TH          (deriveJSON,defaultOptions)
import           Data.Int               (Int32, Int64)
import           Data.List              (genericLength)
import qualified Data.Text.IO           as T
import qualified Data.Vector            as V

import qualified TensorFlow.Core        as TF
import qualified TensorFlow.Ops         as TF
import qualified TensorFlow.Gradient    as TF

import           GHC.Stack
-- import qualified TensorFlow.Minimize    as TF
\end{code}


This module is for the ``nao_classify_server''. Here I use the neural network to classify
whether a human face is ``positive'' or ``negative''.

This neural network has a hidden layer.

\begin{figure}
  \centering
  \begin{tikzpicture}[node distance=3cm,>=stealth',bend angle=20,auto]        
    \tikzstyle{point}=[circle,thick,draw=blue!75,fill=blue!20,minimum size=6mm]
    \begin{scope}
      \node[point](age){age};
      \node[point,below of=age](smile){smile};
      \node[point,below of=smile](gender){gender};
      \node[point,right of=age](h1){}
      edge [pre] node {} (age)
      edge [pre] node {} (smile)
      edge [pre] node {} (gender);
      \node[point,below of=h1](h2){}
      edge [pre] node {} (age)
      edge [pre] node {} (smile)
      edge [pre] node {} (gender);
      \node[point,below of=h2](h3){}
      edge [pre] node {} (age)
      edge [pre] node {} (smile)
      edge [pre] node {} (gender);
      \node[point,right of=h1](y1){}
      edge [pre] node {} (h1)
      edge [pre] node {} (h2)
      edge [pre] node {} (h3);
      \node[point,right of=h2](y2){}
      edge [pre] node {} (h1)
      edge [pre] node {} (h2)
      edge [pre] node {} (h3);
    \end{scope}
  \end{tikzpicture}
  \caption{Neural Network}
  \label{fig:nn}
\end{figure}

\begin{code}
type Label = Int32

data TrainModel = TrainModel { train :: TF.TensorData Float -- ^ [age/100,gender,smile]
                                     -> TF.TensorData Label -- ^ label
                                     -> TF.Session ()
                             , infer :: TF.TensorData Float         -- ^ [age/100,gender,smile]
                                     -> TF.Session (V.Vector Label) -- ^ label
                             , errRt :: TF.TensorData Float -- ^ [age/100,gender,smile]
                                     -> TF.TensorData Label  -- ^ label
                                     -> TF.Session Float
                             , param :: TF.Session PredictModelParam
                             }
data PredictModelParam = PredictModelParam [Float] [Float] [Float] [Float]
                         deriving Show
deriveJSON defaultOptions ''PredictModelParam
data PredictModel = PredictModel { doPredict :: TF.TensorData Float            -- ^ [age/100,gender,smile]
                                             -> TF.Session (V.Vector Label) -- ^ Label
                                 }
-- | Create tensor with random values where the stddev depends on the width.
randomParam :: Int64 -> TF.Shape -> TF.Build (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) =
    (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

reduceMean :: TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float
reduceMean xs = TF.mean xs (TF.scalar (0 :: Int32))

tupleSize :: Int64
tupleSize = 3
labelSize :: Int64
labelSize = 2

commonModel batchSize nodeSize = do
  face <- TF.placeholder [batchSize, tupleSize]
  -- hidden layer
  hiddenWeights <- TF.initializedVariable =<< randomParam tupleSize [tupleSize, nodeSize]
  hiddenBiases  <- TF.zeroInitializedVariable [nodeSize]
  let hiddenZ = (face `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
      hidden  = TF.relu hiddenZ
  -- logits
  logitWeights <- TF.initializedVariable =<< randomParam nodeSize [nodeSize,labelSize]
  logitBiases  <- TF.zeroInitializedVariable [labelSize]
  let logits = (hidden `TF.matMul` logitWeights) `TF.add` logitBiases
  predict <- TF.render $ TF.cast $ TF.argMax (TF.softmax logits) (TF.scalar (1 :: Int32))
  return (face, hiddenWeights, hiddenBiases, logitWeights, logitBiases, predict)


createPredictModel ::  PredictModelParam -> TF.Build PredictModel
createPredictModel (PredictModelParam hw hb lw lb )  = do
  let batchSize = -1 -- use -1 batch size to support variable sized batches
      nodeSize = 3 -- (age, gender, smile) =>> (h1, h2, h3) -> (true, false)
  face <- TF.placeholder [batchSize, tupleSize]
  -- hidden layer
  hiddenWeights <- TF.initializedVariable $ TF.constant [tupleSize,nodeSize] hw
  hiddenBiases  <- TF.initializedVariable $ TF.constant [nodeSize]           hb
  let hiddenZ = (face `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
      hidden  = TF.relu hiddenZ
  -- logits
  logitWeights <- TF.initializedVariable $ TF.constant [nodeSize,labelSize] lw
  logitBiases  <- TF.initializedVariable $ TF.constant [labelSize]          lb
  let logits = (hidden `TF.matMul` logitWeights) `TF.add` logitBiases
  predict <- TF.render $ TF.cast $ TF.argMax (TF.softmax logits) (TF.scalar (1 :: Int32))
  return PredictModel { doPredict = \fFeed -> TF.runWithFeeds [ TF.feed face fFeed] predict
                      }
  

createTrainModel :: HasCallStack => TF.Build TrainModel
createTrainModel = do
  let batchSize = -1 -- use -1 batch size to support variable sized batches
      nodeSize = 3 -- (age, gender, smile) =>> (h1, h2, h3) -> (true, false)
  face <- TF.placeholder [batchSize, tupleSize]
  -- hidden layer
  hiddenWeights <- TF.initializedVariable =<< randomParam tupleSize [tupleSize, nodeSize]
  hiddenBiases  <- TF.zeroInitializedVariable [nodeSize]
  let hiddenZ = (face `TF.matMul` hiddenWeights) `TF.add` hiddenBiases
      hidden  = TF.relu hiddenZ
  -- logits
  logitWeights <- TF.initializedVariable =<< randomParam nodeSize [nodeSize,labelSize]
  logitBiases  <- TF.zeroInitializedVariable [labelSize]
  let logits = (hidden `TF.matMul` logitWeights) `TF.add` logitBiases
  predict <- TF.render $ TF.cast $ TF.argMax (TF.softmax logits) (TF.scalar (1 :: Int32))
  return (face, hiddenWeights, hiddenBiases, logitWeights, logitBiases, predict)
  -- create training action
  labels <- TF.placeholder [batchSize]
  let labelVecs = TF.oneHot labels (fromIntegral labelSize) 1 0
      loss      = reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
      params    = [hiddenWeights, hiddenBiases, logitWeights, logitBiases]
  grads <- TF.gradients loss params
  let lr = TF.scalar 0.00001
      applyGrad param grad = TF.assign param $ param `TF.sub` (lr `TF.mul` grad)
  trainStep <- TF.group =<< zipWithM applyGrad params grads
  let correctPreds = TF.equal predict labels
  errorRateTensor <- TF.render $ 1 - reduceMean (TF.cast correctPreds)
  return TrainModel { train = \fFeed lFeed -> TF.runWithFeeds_
                                              [ TF.feed face   fFeed
                                              , TF.feed labels lFeed
                                              ] trainStep
                    , infer = \fFeed -> TF.runWithFeeds [ TF.feed face fFeed] predict
                    , errRt = \fFeed lFeed -> TF.unScalar <$> TF.runWithFeeds
                                              [ TF.feed face   fFeed
                                              , TF.feed labels lFeed
                                              ] errorRateTensor
                    , param = do
                        hw <- TF.run =<< TF.render (TF.cast hiddenWeights)
                        hb <- TF.run =<< TF.render (TF.cast hiddenBiases)
                        lw <- TF.run =<< TF.render (TF.cast logitWeights)
                        lb <- TF.run =<< TF.render (TF.cast logitBiases)
                        return $ PredictModelParam (V.toList hw) (V.toList hb) (V.toList lw) (V.toList lb)
                    }

training :: HasCallStack
            => [V.Vector Float] -- ^ face  for training
            -> [Label] -- ^ label for training
            -> Int     -- ^ batch size
            -> Int     -- ^ loop limit
            -> IO PredictModelParam

training trFaces trLabels batchSize k = TF.runSession $ do
  model <- TF.build createTrainModel
  let encodeFaceBatch  xs = TF.encodeTensorData [genericLength xs, tupleSize] (mconcat xs)
      encodeLabelBatch xs = TF.encodeTensorData [genericLength xs] (fromIntegral <$> V.fromList xs)
      selectBatch    i xs = take batchSize $ drop (i * batchSize) (cycle xs)
  -- training
  forM_ ([0..k] :: [Int]) $ \i -> do
    let face   = encodeFaceBatch  $ selectBatch i trFaces
        labels = encodeLabelBatch $ selectBatch i trLabels
    train model face labels
    when (i `mod` 100 == 0) $ do
      err <- errRt model face labels
      liftIO $ putStrLn $ "training error " ++ show (err * 100)
  param model

runPredict :: PredictModelParam
           -> [V.Vector Float] -- ^ face for predict
           -> IO [Label]       -- ^ label
runPredict pmp fs = TF.runSession $ do
  pModel <- TF.build $ createPredictModel pmp
  let face = TF.encodeTensorData [genericLength fs, tupleSize] (mconcat fs)
  V.toList <$> doPredict pModel face
\end{code}