module Main where

import           AI.DecisionTree

import           Data.Aeson
import qualified Data.ByteString.Lazy       as BL
import qualified Data.ByteString.Lazy.Char8 as BLC8
import           System.IO

main :: IO ()
main = do
  all <- BL.hGetContents stdin
  let rt = decode'lim all -- :: Maybe ([Attr Int],[Labeled Int Bool])
  case rt of
    Just (attrs,labeleds) ->
      BLC8.putStrLn $ encode $ build'lim attrs labeleds
    x -> hPutStrLn stderr "invalid datas" >> print x
  where build'lim :: [Attr Int] -> [Labeled Int Bool] -> DecisionTree Int Bool
        build'lim = build
        decode'lim :: BL.ByteString -> Maybe ([Attr Int],[Labeled Int Bool])
        decode'lim = decode
