{-# LANGUAGE TemplateHaskell #-}

module Toy.Backend.Regress where

import           AI.Regress.QuadraticCurve
import           Data.Aeson
import           Data.Aeson.TH
import           Data.Text                 (pack)
import           Yesod.Core

data Regress = Regress () deriving Show
deriveJSON defaultOptions ''Regress
regress :: Regress -> [Float] -> [Float] -> IO [Float]
regress _ yaws pitches = do
  print yaws
  print pitches
  w'1:w'2:w'3:w'4:w'5:_ <- fit 1000 1 yaws pitches
  return [w'1,w'2,w'3,w'4,w'5]
