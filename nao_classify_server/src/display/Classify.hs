{-# LANGUAGE TemplateHaskell #-}


module Classify where

import           Data.Aeson
import           Data.Aeson.TH
import           Data.Text     (pack)
import           Yesod.Core

data Classify = Classify () deriving Show
deriveJSON defaultOptions ''Classify
classify :: Classify -> Int -> Bool -> Bool -> IO Bool
classify _ age smile gender= do
  print (age,smile,gender)
  return True
