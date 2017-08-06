\subsubsection{Quadratic Curve Regression}
\label{code:regress:qcr}

\begin{code}
module AI.Regress.QuadraticCurve
  ( fit
  ) where

import           AI.Regress.Internal
\end{code}

In This section, we use $\phi(x,y) = (x^2,y^2,xy,x,y)^T$ to help fitting the model
\[
  b = \mathbf{w}\phi(x,y)
\]


\begin{code}
fit :: Int -> Float -> [Float] -> [Float] -> IO [Float]
fit times v xs' ys' =
  let xs = zipWith (\x y -> [x^2,y^2,x*y,x,y]) xs' ys'
      ys = replicate (length xs') v
  in fit'linear times 5 xs ys
\end{code}
