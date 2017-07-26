module AI.Regress.QuadraticCurve
  ( fit
  ) where


import           AI.Regress.Internal

fit :: Int -> Float -> [Float] -> [Float] -> IO [Float]
fit times v xs' ys' =
  let xs = zipWith (\x y -> [x^2,y^2,x*y,x,y]) xs' ys'
      ys = replicate (length xs') v
  in fit'linear times 5 xs ys

