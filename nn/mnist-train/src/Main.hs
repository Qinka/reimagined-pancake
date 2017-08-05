
module Main where

import           AI.MNIST
import           AI.MNIST.Parse
import           System.Environment

main :: IO ()
main = do
  path'prefix:outfile:times:_ <- getArgs
  s  <- readMNISTSamples $ path'prefix ++ "train-images-idx3-ubyte.gz"
  l  <- readMNISTLabels  $ path'prefix ++ "train-labels-idx1-ubyte.gz"
  ts <- readMNISTSamples $ path'prefix ++  "t10k-images-idx3-ubyte.gz"
  tl <- readMNISTLabels  $ path'prefix ++  "t10k-labels-idx1-ubyte.gz"
  x <- train_mnist (read times) s l ts tl
  writeFile outfile $ show x
