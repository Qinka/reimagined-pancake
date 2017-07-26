{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists  #-}
{-# LANGUAGE TemplateHaskell  #-}

module AI.Regress.Internal
  ( fit'linear
  ) where


import           Control.Monad          (replicateM_)
import           Control.Monad.IO.Class (liftIO)
import           Data.Int               (Int32, Int64)
import           Data.List              (genericLength)
import qualified Data.Vector            as V
import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TF
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF hiding (initializedVariable,
                                               zeroInitializedVariable)
import qualified TensorFlow.Variable    as TF

fit'linear :: Int -> Int64 -> [[Float]] -> [Float] -> IO [Float]
fit'linear times v xs' ys' = TF.runSession $ do
  let xs = TF.constant [genericLength xs',v] $ mconcat xs'
      ys = TF.constant [genericLength ys',1] ys'
  w <- TF.zeroInitializedVariable [v,1]
  let yHat = (xs `TF.matMul` TF.readValue w)
      loss = TF.reduceMean $  TF.square (yHat `TF.sub` ys)
  trainStep <- TF.minimizeWith TF.adam loss [w]
  replicateM_ times (TF.run trainStep)
  V.toList <$> (TF.run =<< TF.render (TF.cast $ TF.readValue w))
