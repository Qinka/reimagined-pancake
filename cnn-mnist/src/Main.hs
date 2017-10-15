{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveDataTypeable #-}

module Main where

import Parser.MNIST
import CNN

import System.Console.CmdArgs

data Args = Args
            { aSaveFile :: FilePath
            , aRestore  :: Bool
            , aTrainImage :: FilePath
            , aTrainLabel :: FilePath
            , aTestImage :: FilePath
            , aTestLabel :: FilePath
            , aTimes     :: Int
            }
            deriving (Show,Data,Typeable)

cnn :: Args
cnn = Args
  { aSaveFile = ".save/save"
    &= typFile
    &= help "where to store"
    &= explicit
    &= name "save-file"
    &= name "s"
  , aRestore = False
    &= help "whether try to restore"
    &= explicit
    &= name "restore"
    &= name "r"
  , aTrainImage = "train-images-idx3-ubyte.gz"
    &= typFile
    &= help "Training data(images)"
    &= explicit
    &= name "train-image"
    &= name "i"
  , aTrainLabel = "train-labels-idx1-ubyte.gz"
    &= typFile
    &= help "Training data(labels)"
    &= explicit
    &= name "train-label"
    &= name "l"
  , aTestImage = "t10k-images-idx3-ubyte.gz"
    &= typFile
    &= help "Testing data(image)"
    &= explicit
    &= name "test-image"
    &= name "t"
  , aTestLabel = "t10k-labels-idx1-ubyte.gz"
    &= typFile
    &= help "Testing data(label)"
    &= explicit
    &= name "test-label"
    &= name "T"
  , aTimes = 10
    &= typ "INT"
    &= explicit
    &= name "time"
  }

main = do
  Args{..} <- cmdArgs cnn
  trainImage <- readMNISTSamples aTrainImage
  trainLabel <- readMNISTLabels  aTrainLabel
  testImage  <- readMNISTSamples aTestImage
  testLabel  <- readMNISTLabels  aTestLabel
  runMain trainImage trainLabel testImage testLabel aSaveFile aRestore aTimes
