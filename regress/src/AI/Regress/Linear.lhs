\subsubsection{Linear Regression}
\label{code:regress:lr}

\begin{code}
module AI.Regress.Linear
  ( fit
  ) where

import           AI.Regress.Internal
\end{code}

This section use the multiple linear regression to regress model like $y=wx+b$.
The real model fitted is
\[
  b = w_xx+w_yy
\]

\begin{code}
fit :: Int -> Float -> [Float] -> [Float] -> IO [Float]
fit times v xs' ys' =
  let xs = zipWith xy xs' ys'
      ys = replicate (length xs') v
      xy x y = [x,y]
  in fit'linear times 2 xs ys
\end{code}
