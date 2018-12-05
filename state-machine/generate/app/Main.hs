{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Text.Prettyprint.Doc
import Data.Text.Prettyprint.Doc.Render.Text

import qualified StateMachine.Pretty as P
import qualified System.IO           as IO

{-# ANN module ("HLint: ignore Redundant do" :: String) #-}

withCHeader :: Doc () -> Doc ()
withCHeader header = vsep
  [ "#include <stdint.h>"
  , "#include <immintrin.h>"
  , ""
  , header
  ]

main :: IO ()
main = do
  IO.withFile "../common/transition-table.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.transitionTable)
  IO.withFile "../common/phi-table.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.phiTable)
  IO.withFile "../common/simd-transition-table-32.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdTransitionTable32)
  IO.withFile "../common/simd-phi-table-32.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdPhiTable32)
  IO.withFile "../common/simd-transition-table-128.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdTransitionTable128)
  IO.withFile "../common/simd-phi-table-128.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdPhiTable128)
  IO.withFile "../common/simd-transition-phi-table.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdTransitionPhiTable128)
  IO.withFile "../common/simd-transition-phi-wide-table.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdTransitionPhiTable256)
  IO.withFile "../common/simd-transition-table-saturated-256.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdTransitionTableSaturated256)
  IO.withFile "../common/simd-transition-phi-table-saturated-256.c" IO.WriteMode $ \h ->
    hPutDoc h (withCHeader P.simdTransitionPhiTableSaturated256)
