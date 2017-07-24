{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TemplateHaskell #-}
module Toy.Backend.Classify where

import           AI.NN
import           Data.Aeson
import           Data.Aeson.TH
import           Data.Text     (pack)
import qualified Data.Vector   as V
import           Yesod.Core

data Classify = Classify PredictModelParam deriving Show
deriveJSON defaultOptions ''Classify

classify :: Classify -> Int -> Bool -> Bool -> IO Bool
classify (Classify pmp) age smile gender = do
  print (age,smile,gender)
  let a = fromIntegral age / 100
      s = if smile  then 1 else 0
      g = if gender then 1 else 0
      faces = [[a,g,s]]
  rt <- runPredict pmp faces
  print rt
  return $ not $ head rt == 0
