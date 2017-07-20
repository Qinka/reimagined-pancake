
\section{K-Nearest Neighbor}
\label{sec:knn}
\begin{code}
{-# LANGUAGE RecordWildCards  #-}
{-# LANGUAGE TypeOperators    #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TemplateHaskell  #-}

module AI.KNN
       ( decide
       , KNNStore(..)
       ) where

import           Data.Aeson.TH
import           Data.Array.Accelerate ((!),(:.)(..))
import qualified Data.Array.Accelerate as A
import           Data.List
\end{code}

In this section, the topic is about k-nearest neighbor algorithm. 
The basic idea of this algorithm is simple: find out the k-nearest neighbor of one point,
and then label this point with its neighbors' label.

\subsection{ADT}
\label{sec:knn:adt}

So the first thing is ``ADT''.
\begin{code}
data KNNStore a b c = KNNStore
                      { knnRow   ::  a    -- ^ row
                      , knnCol   ::  a    -- ^ col
                      , knnData  :: [b]   -- ^ data
                      , knnLabel :: [c]   -- ^ labels
                      , knnK     :: Int   -- ^ K
                      }
                  deriving Show
deriveJSON defaultOptions ''KNNStore
\end{code}

\subsection{Prediction}
\label{sec:knn:pd}

We use \lstinline|KNNStore| and data for prediction.

Translate KNNStore into ``accelerate'' format so that we can compute the distance.
\begin{code}
knnTransAcc :: (A.Elt b, Integral a)
               => KNNStore a b c    -- ^ stored data
               -> A.Array A.DIM2 b  -- ^ matrix
knnTransAcc KNNStore{..} = A.fromList (A.Z :. fromIntegral knnRow :. fromIntegral knnCol) knnData
\end{code}

Translate the testing data into ``accelerate'' format.
\begin{code}
predTransAcc :: (A.Elt b, Integral a)
                => KNNStore a b c    -- ^ ``settings''
                -> [b]               -- ^ testing data
                -> A.Array A.DIM1 b  -- ^ vector
predTransAcc KNNStore{..} dat = A.fromList (A.Z :. fromIntegral knnCol) dat
\end{code}

Compute the difference between stored points and testing point.
\begin{code}
computeDiff :: (A.Elt b, A.Num b)
               => A.Acc (A.Array A.DIM2 b) -- ^ stored points
               -> A.Acc (A.Array A.DIM1 b) -- ^ testing point
               -> A.Acc (A.Array A.DIM2 b) -- ^ result
computeDiff sps tp = A.generate (A.shape sps) opt
  where A.Z :. row :. col = A.unlift (A.shape sps) :: A.Z :. A.Exp Int :. A.Exp Int
        opt sh = let A.Z :. r :. c = A.unlift sh :: A.Z :. A.Exp Int :. A.Exp Int
                     sh' = A.lift $ A.Z :. c :: A.Exp A.DIM1
                 in (sps ! sh) - (tp ! sh')
\end{code}

Compute the distance via folding
\begin{code}
computeDistance :: (A.Elt b, A.Floating b)
                   => A.Acc (A.Array A.DIM2 b) -- ^ differences
                   -> A.Acc (A.Array A.DIM1 b) -- ^ distance
computeDistance = A.map A.sqrt . A.fold opt 0
  where opt i p = p + i ^ 2
\end{code}

\begin{code}
decide :: (Integral a, Floating b, A.Floating b, Ord b, A.Elt b, Ord c)
          => (A.Acc (A.Array A.DIM1 b)
              -> (A.Array A.DIM1 b)) -- ^ run
          -> KNNStore a b c     -- ^ stored data
          -> [b]                -- ^ testing data
          -> c                  -- ^ label
decide run k@KNNStore{..} t =
  snd $ head $ sort $ map (\x -> (- length x,head x)) $ group $ map snd $ take knnK $ sort $ zip dis knnLabel
  where sd = knnTransAcc  k
        td = predTransAcc k t
        disOpt sps = computeDistance . computeDiff sps
        dis = A.toList $ run $ disOpt (A.use sd) (A.use td)
\end{code}
