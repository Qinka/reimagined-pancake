module Main where

import           AI.DecisionTree

import           Data.Aeson
import qualified Data.ByteString.Lazy       as BL
import qualified Data.ByteString.Lazy.Char8 as BLC8
import           System.IO

main :: IO ()
main = do
  let attr = [Attr "asd" [1,3,5],Attr "vcv" [2,4,6]]
  let labeled = [(True,Unlabeled "a" (zip attr [1,4]))]
  BLC8.hPutStrLn stderr $ encode ((attr,labeled) :: ([Attr Int],[Labeled Int Bool]))
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
