-- NOTE: This source is a modified copy of
-- https://github.com/tensorflow/haskell/blob/master/tensorflow-mnist/src/TensorFlow/Examples/MNIST/Parse.hs
-- and this module is not with GPLv3+ but Apache, while the Google will be the one who own the copyright.
-- The copyright of the codes modified are with GPLv3+, and I(the coder) will has the copyright.
-- /--| I(coder) am not goot at English, so the aboves or followings might not express myself right.

-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.


{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE OverloadedLists      #-}
{-# LANGUAGE OverloadedStrings    #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE ViewPatterns         #-}

module AI.MNIST.Parse {-# WARNING "This model is just a copy from haskell's tensorflow binding, reading the text at top of this module's codes"#-}
where

import qualified Codec.Compression.GZip as GZip
import           Control.Monad          (liftM, when)
import           Data.Binary.Get        (Get, getLazyByteString, getWord32be,
                                         runGet)
import           Data.ByteString.Lazy   (readFile, toStrict)
import qualified Data.ByteString.Lazy   as L
import           Data.List.Split        (chunksOf)
import           Data.Monoid            ((<>))
import           Data.ProtoLens         (Message, decodeMessageOrDie)
import           Data.Text              (Text)
import qualified Data.Text              as Text
import qualified Data.Vector            as V
import           Data.Word              (Word32, Word8)
import           Prelude                hiding (readFile)

-- | Utilities specific to MNIST.
type MNIST = V.Vector Word8

-- | Produces a unicode rendering of the MNIST digit sample.
drawMNIST :: MNIST -> Text
drawMNIST = chunk . block
  where
    block :: V.Vector Word8 -> Text
    block (V.splitAt 1 -> ([0], xs)) = " " <> block xs
    block (V.splitAt 1 -> ([n], xs)) = c `Text.cons` block xs
      where c = "\9617\9618\9619\9608" !! fromIntegral (n `div` 64)
    block (V.splitAt 1 -> _)   = ""
    chunk :: Text -> Text
    chunk "" = "\n"
    chunk xs = Text.take 28 xs <> "\n" <> chunk (Text.drop 28 xs)

-- | Check's the file's endianess, throwing an error if it's not as expected.
checkEndian :: Get ()
checkEndian = do
    magic <- getWord32be
    when (magic `notElem` ([2049, 2051] :: [Word32])) $
        fail "Expected big endian, but image file is little endian."

-- | Reads an MNIST file and returns a list of samples.
readMNISTSamples :: FilePath -> IO [MNIST]
readMNISTSamples path = do
    raw <- GZip.decompress <$> readFile path
    return $ runGet getMNIST raw
  where
    getMNIST :: Get [MNIST]
    getMNIST = do
        checkEndian
        -- Parse header data.
        cnt  <- liftM fromIntegral getWord32be
        rows <- liftM fromIntegral getWord32be
        cols <- liftM fromIntegral getWord32be
        -- Read all of the data, then split into samples.
        pixels <- getLazyByteString $ fromIntegral $ cnt * rows * cols
        return $ V.fromList <$> chunksOf (rows * cols) (L.unpack pixels)

-- | Reads a list of MNIST labels from a file and returns them.
readMNISTLabels :: FilePath -> IO [Word8]
readMNISTLabels path = do
    raw <- GZip.decompress <$> readFile path
    return $ runGet getLabels raw
  where getLabels :: Get [Word8]
        getLabels = do
            checkEndian
            -- Parse header data.
            cnt <- liftM fromIntegral getWord32be
            -- Read all of the labels.
            L.unpack <$> getLazyByteString cnt

readMessageFromFileOrDie :: Message m => FilePath -> IO m
readMessageFromFileOrDie path = do
    pb <- readFile path
    return $ decodeMessageOrDie $ toStrict pb
