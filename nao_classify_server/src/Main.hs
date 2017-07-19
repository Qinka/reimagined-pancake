{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes       #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE TypeFamilies      #-}

module Main where
-- classifty :: (MonadIO m,ToJSON Classifty) => Classifty -> a -> b -> c -> m Bool
import           Classify                   (Classify (..), classify)

import           Data.Aeson
import           Data.Aeson.TH
import qualified Data.ByteString.Lazy       as BL
import qualified Data.ByteString.Lazy.Char8 as BLC8
import           Data.Text                  (Text)
import qualified Data.Text                  as T
import           System.IO
import           Yesod.Core

data NaoClassifyServer = NaoClassifyServer Int Classify
                         deriving Show
deriveJSON defaultOptions ''NaoClassifyServer

mkYesod "NaoClassifyServer" [parseRoutes|/classify ClassifyR GET|]
instance Yesod NaoClassifyServer

getClassifyR :: Handler Value
getClassifyR = do
  age    <- lookupGetParam "age"
  smile  <- lookupGetParam "smile"
  gender <- lookupGetParam "gender"
  case (age,smile,gender) of
    (Just a,Just s,Just g) -> do
      NaoClassifyServer _ cl <- getYesod
      v <- liftIO $ classify cl (readT a) (readT s) (readT g)
      returnJson v
    x -> do
      $logError (T.pack $ show x)
      invalidArgs ["need more args",T.pack $ show x]

main :: IO ()
main = BLC8.getContents >>=
  (\cfg -> case decode cfg of
      Just c@(NaoClassifyServer p _) -> warp p c
      _                              -> hPutStrLn stderr "invalid config"
  )
readT :: Read a => Text -> a
readT = read . T.unpack
