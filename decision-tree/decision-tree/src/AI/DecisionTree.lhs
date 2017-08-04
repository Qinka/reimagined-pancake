
\subsubsection{Decision Tree}
\label{sec:dt:code:dt}

So in this section, the topic is about decision tree. The tree will be learning via ID3 algorithm.
\begin{code}
{-# LANGUAGE RecordWildCards #-}
module AI.DecisionTree
       ( build
       , decide
       , Unlabeled(..)
       , Labeled(..)
       , Attr(..)
       , DecisionTree(..)
       ) where

import           Data.Function       (on)
import           Data.List           hiding (partition)
import           Data.Map            (Map)
import qualified Data.Map            as Map
import           Data.Maybe          (fromJust)
import           AI.DecisionTree.ADT
\end{code}

Build the decision tree
\begin{code}
build :: (Ord a, Ord b)
         => [Attr a]    -- ^ the attribute
         -> [Labeled a b]    -- ^ labeled data (for training)
         -> DecisionTree a b -- ^ the tree
build attrs labeledset =  case inf of
    0 -> DTLeaf dominantLabel -- Even if the best Attribute doesn't gain any information, we're done
    _ -> DTNode { dtnAttr = bestAttr
                , dtnKids = allKids
                }
  where  (inf,bestAttr) = bestAttribute labeledset attrs -- get the best attribute
         p              =  partition labeledset bestAttr -- use it to partition the set
         allKids        = Map.map (build attrs) p -- recursivly build the children
         dominantLabel  = fst $ Map.findMax $ groupLabels (map label labeledset) -- in case we are done, get the label
\end{code}

To get the value that the labeled data has for given attribute
\begin{code}
getValue :: Unlabeled a -- ^ unlabled one
         -> Attr      a -- ^ attribute
         ->           a -- ^ value
getValue Unlabeled{..} attr = fromJust $ lookup attr uAttributes
\end{code}

Return a labeled one's label
\begin{code}
label :: Labeled a b -- ^ labeled one
      -> b           -- ^ label
label = fst
\end{code}

Find out which label belongs to the unlabeled set
\begin{code}
decide :: (Ord a, Eq a)
          => DecisionTree a b -- ^ decision tree
          -> Unlabeled    a   -- ^ unlabeled on
          ->                b 
decide (DTLeaf b)  _        = b                            -- we reached a Leaf, done
decide  DTNode{..} unlabeled = decide (dtnKids `safeLookup` v) unlabeled -- we're in a node, walk down
  where v = getValue unlabeled dtnAttr
\end{code}

Partition the labeled set by the possible values of the attribute
\begin{code}
partition :: (Ord a)
             => [Labeled a b] -- ^ labeled items
             -> Attr a        -- ^ attribute
             -> Map a [Labeled a b] -- ^ output
partition set attr@Attr{..} = foldl (\m k -> Map.insertWith (++) k [] m) grouped aValues
  where grouped = groupWith (flip getValue attr . snd) (:[]) (++) set
\end{code}

For entropy
\begin{equation}
  \label{eq:entropy}
  entropy = \sum\limits_{i \in \{\text{has label i}\}} - \dot{p}_i * \log_2 \dot{p}_i
\end{equation}
\begin{code}
entropy :: (Ord b)
           => [b]    -- ^ labels
           -> Double -- ^ value of entropy
entropy set = negate $ Map.fold helper 0 $ groupLabels set 
    where n = fromIntegral $ length set
          helper 0 _   = error "entropy: we are not supposed to get p=0"
          helper s acc = let p = fromIntegral s / n in acc + p * logBase 2 p
\end{code}

Count how mant data we have for each label, then group it with 1 as beginner.
\begin{code}
groupLabels :: Ord b
               => [b]        -- ^ labels
               -> Map b Int  -- ^ mapped labels
groupLabels = groupWith id (const (1::Int)) (const succ)
\end{code}

For information:
\begin{equation}
  \label{eq:information}
  info = entropy(set) - \sum \dot{p}_i * entropy(dat_i | dat \in \text{has value i for attr a})
\end{equation}
\begin{code}
information :: (Ord b, Ord a)
               => [Labeled a b] -- ^ the data
               -> Attr a        -- ^ the attribute
               -> Double        -- ^ the information
information labeled attr = entropy (map label labeled) - sum (zipWith (*) p'i $ map entropy ps)
  where ps  = map (map label) $ Map.elems $ partition labeled attr -- the partitions, we're only interested in the labels
        p'i = map ((/n) . fromIntegral . length) ps -- the size of the partition/size labeledset
        n   = fromIntegral $ length labeled
\end{code}

Get the attribute whose informatio is greatest (gain in)
\begin{code}
bestAttribute :: (Ord b, Ord a)
                 => [Labeled a b]   -- ^ labeled data
                 -> [Attr a]        -- ^ attributes
                 -> (Double,Attr a)
bestAttribute lss = head.sortBy (compare `on` negatedInf).computeInformation
  where negatedInf (inf,_) = -inf
        computeInformation = map (\x -> (information lss x,x))
\end{code}

Groups a labeled set using a Map. According to Haskell ``efficient'' grouping needs Ord.
\begin{code}
groupWith :: Ord k
             => (a -> k)      -- ^ how to extract keys from unlabeleds
             -> (a -> v)      -- ^ how to make unlabeled into values for the map
             -> (v -> v -> v) -- ^ how to fuse two values (should we have > 1 data for this key)
             -> [a]           -- ^ the list we want to group
             -> Map k v
groupWith getKey singleton fuse =
    foldl (\m x -> Map.insertWith fuse (getKey x) (singleton x) m) Map.empty
\end{code}



