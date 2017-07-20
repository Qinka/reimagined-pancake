{-# LANGUAGE TemplateHaskell #-}
module Classify where

import           Accelerate
import           AI.KNN
import           Data.Aeson
import           Data.Aeson.TH
import           Data.Text     (pack)
import           Yesod.Core

data Classify = Classify (KNNStore Int Float Bool) deriving Show
deriveJSON defaultOptions ''Classify

classify :: Classify -> Int -> Bool -> Bool -> IO Bool
classify (Classify knn) age smile gender = do
  print (age,smile,gender)
  return $ decide run knn [fromIntegral age, fromIntegral (fromEnum smile),fromIntegral (fromEnum gender)]
