{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes       #-}
{-# LANGUAGE RecordWildCards   #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TypeFamilies      #-}

module Main where
-- classifty :: (MonadIO m, FromJSON Classifty) => Classifty -> a -> b -> c -> m Bool
import           Toy.Backend.Classify       (Classify (..), classify)
-- regress :: (MonadIO m, FromJSON Regress) => Regress -> [a] -> [a] -> m [a]
import           Toy.Backend.Regress        (Regress (..), regress)

import           Data.Aeson
import           Data.Aeson.TH
import qualified Data.ByteString.Lazy       as BL
import qualified Data.ByteString.Lazy.Char8 as BLC8
import           Data.Text                  (Text)
import qualified Data.Text                  as T
import           System.IO
import           Yesod.Core

data ToyBackend = ToyBackend { port       :: Int -- ^ port
                             , classifier :: Classify -- ^ classify config
                             , regressor  :: Regress -- ^ regress config
                             }
                         deriving Show
deriveJSON defaultOptions ''ToyBackend

mkYesod "ToyBackend" [parseRoutes|/classify ClassifyR GET
                                  /regress  RegressR  GET
                                  |]
instance Yesod ToyBackend

main :: IO ()
main = BLC8.getContents >>=
  (\cfg -> case decode cfg of
      Just c@ToyBackend{..} -> warp port c
      _                     -> hPutStrLn stderr "invalid config"
  )


readT :: Read a => Text -> a
readT = read . T.unpack


getClassifyR :: Handler Value
getClassifyR = do
  age    <- lookupGetParam "age"
  smile  <- lookupGetParam "smile"
  gender <- lookupGetParam "gender"
  case (age,smile,gender) of
    (Just a,Just s,Just g) -> do
      cl <- classifier <$> getYesod
      v <- liftIO $ classify cl (readT a) (readT s) (readT g)
      returnJson v
    x -> do
      $logError (T.pack $ show x)
      invalidArgs ["need more args",T.pack $ show x]

getRegressR :: Handler Value
getRegressR = do
  pitches <- lookupGetParams "pitch"
  yaws    <- lookupGetParams "yaw"
  rg      <- regressor <$> getYesod
  returnJson =<< (liftIO $ regress rg (readT <$> yaws) (readT <$> pitches))
