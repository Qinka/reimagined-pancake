
import           System.IO
import           Text.Parsec

parsecLog :: Parsec String () (Int,Float)
parsecLog = do
  spaces
  x <- many1 digit
  skipMany (oneOf "/0123456789")
  skipMany (noneOf "0123456789")
  y <- many1 (noneOf "%") <* char '%'
  spaces
  return (read x, read y)

printLog :: (Int,Float) -> IO ()
printLog (x,y) = putStrLn $ show x ++ " " ++ show y

main = do
  log <- getContents
  case runP (many parsecLog) () "stdin" log of
    Right plot -> mapM_ printLog plot
    Left e     -> hPutStrLn stderr $ show e

