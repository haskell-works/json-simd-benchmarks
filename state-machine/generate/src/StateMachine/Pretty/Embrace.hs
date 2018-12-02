{-# LANGUAGE OverloadedStrings #-}

module StateMachine.Pretty.Embrace where

import Data.List
import Data.Text.Prettyprint.Doc
import Prelude                   hiding ((<>))

embrace :: [Doc ()] -> Doc ()
embrace (d:ds) = intro "{" d <> mconcat (intro "," <$> ds) <> "}"
  where intro p e = p <> " " <> nest 2 e <> line
embrace [] = "{}"

embraceN :: [[Doc ()]] -> Doc ()
embraceN (ds:dss) = vsep $ concat
  [ ["{ " <> mconcat (intersperse ", " ds)]
  , (", " <>) . mconcat . intersperse ", " <$> dss
  , ["}"]
  ]
embraceN [] = "{}"
