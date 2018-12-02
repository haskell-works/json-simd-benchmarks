{-# LANGUAGE OverloadedStrings #-}

module StateMachine.Pretty.PLit
  ( PLit(..)
  ) where

import           Data.String
import           Data.Text.Prettyprint.Doc
import           Data.Word
import           Numeric
import           Prelude                   hiding ((<>))
import           StateMachine.List
import           StateMachine.Pretty.WZero
import           StateMachine.Vec

class PLit a where
  plit :: a -> Doc ()

instance PLit Word8 where
  plit w = fromString $ '0':'x':padl 2 '0' (showHex w [])

instance PLit Word32 where
  plit w = fromString $ '0':'x':padl 8 '0' (showHex w [])

instance PLit Word64 where
  plit w = fromString $ '0':'x':padl 16 '0' (showHex w [])

instance (PLit a, PLit b, PLit c, PLit d, PLit e, PLit f, PLit g, PLit h) => PLit (Vec8 a b c d e f g h) where
  plit (Vec8 a b c d e f g h) = "{"
    <> plit a <> ", "
    <> plit b <> ", "
    <> plit c <> ", "
    <> plit d <> ", "
    <> plit e <> ", "
    <> plit f <> ", "
    <> plit g <> ", "
    <> plit h <> "}"

instance (PLit a, PLit b, PLit c, PLit d) => PLit (Vec4 a b c d) where
  plit (Vec4 a b c d) = "{"
    <> plit a <> ", "
    <> plit b <> ", "
    <> plit c <> ", "
    <> plit d <> "}"

instance (PLit a, PLit b) => PLit (Vec2 a b) where
  plit (Vec2 a b) = "{"
    <> plit a <> ", "
    <> plit b <> "}"

instance (PLit WZero) where
  plit _ = "0"
