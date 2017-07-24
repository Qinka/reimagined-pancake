{-# LANGUAGE TemplateHaskell #-}
module Toy.Backend.Classify where

import           AI.DecisionTree
import           Data.Aeson
import           Data.Aeson.TH
import           Data.Text       (pack)
import           Yesod.Core

data Classify = Classify (DecisionTree Bool Bool)  deriving Show
deriveJSON defaultOptions ''Classify

classify :: Classify -> Int -> Bool -> Bool -> IO Bool
classify (Classify dt) age smile gender = do
  print (age,smile,gender)
  return $ decide dt $ Unlabeled "test" $ zip [Attr "age" [], Attr "smile" [], Attr "gender" []] [age > 32 , smile,gender]
