{-# LANGUAGE TemplateHaskell #-}

module Toy.Backend.Regress where

import           AI.Regress.Linear
import           Data.Aeson
import           Data.Aeson.TH
import           Data.Text         (pack)
import           Yesod.Core

data Regress = Regress () deriving Show
deriveJSON defaultOptions ''Regress
regress :: Regress -> [Float] -> [Float] -> IO [Float]
regress _ yaws pitches = do
  print yaws
  print pitches
  (w,b) <- fit 1000 yaws pitches
  return [w,b]
