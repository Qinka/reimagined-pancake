{-# LANGUAGE TemplateHaskell #-}
module Classify where

import           AI.SVM.Base
import           AI.SVM.Simple
import           Data.Aeson
import           Data.Aeson.TH
import           Data.Text     (pack)

data Classify = Classify FilePath Double deriving Show
deriveJSON defaultOptions ''Classify

classify :: Classify -> Int -> Bool -> Bool -> IO Bool
classify (Classify fp threshold) age smile gender = do
  svm <- loadSVM fp
  let b2d :: Bool -> Double
      b2d = \x -> if x then 1 else -1
      prediction =  predict svm [fromIntegral age / 100, b2d gender, b2d smile]
  return $ prediction > threshold

