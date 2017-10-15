{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}

module Parser.MNIST where

import Control.Monad (when, liftM)
import Data.Binary.Get (Get, runGet, getWord32be, getLazyByteString)
import Data.ByteString.Lazy (toStrict, readFile)
import Data.List.Split (chunksOf)
import Data.Monoid ((<>))
import Data.Int (Int32,Int64)
import Data.Text (Text)
import Data.Word (Word8, Word32)
import Prelude hiding (readFile)
import qualified Codec.Compression.GZip as GZip
import qualified Data.ByteString.Lazy as L
import qualified Data.Text as Text
import qualified Data.Vector as V

-- label
type Label = Int32
type Image = Float
-- | Utilities specific to MNIST.
type MNIST = V.Vector Float

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
        return $ V.fromList . normal <$> chunksOf (rows * cols) (L.unpack pixels)

-- | Reads a list of MNIST labels from a file and returns them.
readMNISTLabels :: FilePath -> IO [Label]
readMNISTLabels path = do
    raw <- GZip.decompress <$> readFile path
    return $ runGet getLabels raw
  where getLabels :: Get [Label]
        getLabels = do
            checkEndian
            -- Parse header data.
            cnt <- liftM fromIntegral getWord32be
            -- Read all of the labels.
            map fromIntegral . L.unpack <$> getLazyByteString cnt

normal :: [Word8] -> [Float]
normal = map $ \x -> (fromIntegral x - 128) / 128
