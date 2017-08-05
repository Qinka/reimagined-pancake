\begin{code}
module AI.Regress.Linear
  ( fit
  ) where


import           AI.Regress.Internal

fit :: Int -> Float -> [Float] -> [Float] -> IO [Float]
fit times v xs' ys' =
  let xs = zipWith xy xs' ys'
      ys = replicate (length xs') v
      xy x y = [x,y]
  in fit'linear times 2 xs ys
\end{code}
