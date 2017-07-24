{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists  #-}
{-# LANGUAGE TemplateHaskell  #-}

module AI.Regress.Linear
       ( fit
       ) where


import           Control.Monad          (replicateM_)
import           Control.Monad.IO.Class (liftIO)
import           Data.Int               (Int32, Int64)
import           Data.List              (genericLength)
import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TF
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF hiding (initializedVariable,
                                               zeroInitializedVariable)
import qualified TensorFlow.Variable    as TF

fit :: Int -> [Float] -> [Float] -> IO (Float,Float)
fit times xs' ys' = TF.runSession $ do
  let xs = TF.vector xs'
      ys = TF.vector ys'
  w <- TF.initializedVariable 0
  b <- TF.initializedVariable 0
  let yHat = (xs `TF.mul` TF.readValue w) `TF.add` TF.readValue b
      loss = TF.square (yHat `TF.sub` ys)
  trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
  replicateM_ times (TF.run trainStep)
  (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
  return (w', b')

