\subsubsection{ADT}
\label{sec:dt:code:adt}

So first we will define the ADTs.

\begin{code}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}

module AI.DecisionTree.ADT
       ( Labeled
       , DecisionTree(..)
       , safeLookup
       , showTree
       , Attr(..)
       , Unlabeled(..)
       ) where

import           Data.Aeson
import           Data.Aeson.TH
import           Data.Char
import           Data.Function       (on)
import           Data.Map            (Map)
import qualified Data.Map            as Map
import           Data.Maybe          (fromJust)
import           GHC.Generics
\end{code}

For datas to training, there the \lstinline|Labeled| will be defined to represent this training or testing data.
\begin{code}
type Labeled a b = (b,Unlabeled a)
\end{code}

Then the ADT for decision tree will be there:
\begin{code}
data DecisionTree a b = DTLeaf b                                     -- ^ leaf
                      | DTNode { dtnAttr :: Attr a                   -- ^ attribute
                               , dtnKids :: Map a (DecisionTree a b) -- ^ functino to get next level
                               }
                        deriving (Generic)
\end{code}
Display the tree.
\begin{code}
instance (Ord a,Show a, Show b) => Show (DecisionTree a b) where
    show x = showTree x ""
showTree :: (Ord a,Show a, Show b) => DecisionTree a b -> ShowS
showTree (DTLeaf x) = shows x
showTree (DTNode attr kids) = ('<':).shows attr.("|\n"++).showList [kids `safeLookup` a | a <- aValues attr].('>':)
\end{code}

Safy search
\begin{code}
safeLookup :: Ord k
              => Map k a
              -> k
              -> a
safeLookup kids a = fromJust $ Map.lookup a kids
\end{code}

Define the attribute
\begin{code}
data Attr a = Attr { aAttrName :: String -- ^ the name for attribute
                   , aValues   :: [a]    -- ^ the possible values for attribute
                   }
instance Eq (Attr a) where
    (==) = (==) `on` aAttrName
instance Show (Attr a) where
    show = aAttrName
\end{code}

The data with out label, or say what we will to classify.
\begin{code}
data Unlabeled a = Unlabeled { uInstanceName :: String       -- ^ the name of this instance
                             , uAttributes   :: [(Attr a,a)] -- ^ the attributes of instance
                             } deriving Show
\end{code}



To/From json
\begin{code}
instance (Ord a,FromJSON a, FromJSON b, FromJSONKey a) => FromJSON (DecisionTree a b)
instance (Ord a,ToJSON   a, ToJSON   b, ToJSONKey   a) => ToJSON   (DecisionTree a b)
deriveJSON  defaultOptions ''Attr
deriveJSON  defaultOptions ''Unlabeled
\end{code}
