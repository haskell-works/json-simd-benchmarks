{-# LANGUAGE OverloadedStrings #-}

module StateMachine.Pretty
  ( transitionTable
  , phiTable
  , simdPhiTable32
  , simdTransitionPhiTable128
  , simdTransitionPhiTable256
  , simdTransitionTable
  , simdTransitionTableSaturated256
  , simdTransitionPhiTableSaturated256
  ) where

import Data.List
import Data.List.Split
import Data.Text.Prettyprint.Doc
import Data.Word
import HaskellWorks.Data.Bits.BitWise
import Prelude                        hiding ((<>))
import StateMachine.Id
import StateMachine.Pretty.Embrace
import StateMachine.Pretty.PLit
import StateMachine.Pretty.WZero
import StateMachine.Vec

import qualified Data.Vector          as DV
import qualified Data.Vector.Storable as DVS
import qualified StateMachine.Table   as SM

transitionTable :: Doc ()
transitionTable = vsep
  [ "uint8_t transition_table[][256] ="
  , embrace (mkV <$> wss) <> ";"
  , ""
  ]
  where wss = DVS.toList <$> DV.toList SM.transitionTable
        mkV :: [Word8] -> Doc ()
        mkV ws = embraceN (chunksOf 16 (fmap plit ws))

phiTable :: Doc ()
phiTable = vsep
  [ "uint8_t phi_table[][256] ="
  , embrace (mkV <$> wss) <> ";"
  , ""
  ]
  where wss = DVS.toList <$> DV.toList SM.phiTable
        mkV :: [Word8] -> Doc ()
        mkV ws = embraceN (chunksOf 16 (fmap plit ws))

simdTransitionTable :: Doc ()
simdTransitionTable = vsep
  [ "uint32_t transition_table[] ="
  , embraceN (chunksOf 8 (fmap plit ws))
  , ""
  ]
  where ws = word32 . fromIntegral <$> DVS.toList SM.transitionTableSimd

simdPhiTable32 :: Doc ()
simdPhiTable32 = vsep
  [ "uint32_t phi_table[] ="
  , embraceN (chunksOf 8 (fmap plit ws))
  , ""
  ]
  where ws = word32 . fromIntegral <$> DVS.toList SM.phiTableSimd

simdTransitionPhiTable256 :: Doc ()
simdTransitionPhiTable256 = vsep
  [ "__m256i simd_transition_phi_wide_table[256] ="
  , nest 2 ("  " <> embraceN (chunksOf 1 (fmap plit pts))) <> ";"
  , ""
  ]
  where pts = zipWith (\p t -> Vec4 t WZero p WZero)
                  (word64 . fromIntegral <$>  DVS.toList SM.phiTableSimd        )
                  (                           DVS.toList SM.transitionTableSimd )

simdTransitionPhiTable128 :: Doc ()
simdTransitionPhiTable128 = vsep
  [ "__m128i simd_transition_phi_table[256] ="
  , nest 2 ("  " <> embraceN (chunksOf 1 (fmap plit pts))) <> ";"
  , ""
  ]
  where pts = zipWith (flip Vec2)
                  (word64 . fromIntegral <$>  DVS.toList SM.phiTableSimd        )
                  (                           DVS.toList SM.transitionTableSimd )

simdTransitionTableSaturated256 :: Doc ()
simdTransitionTableSaturated256 = vsep
  [ "__m256i simd_transition_saturated_256[256] ="
  , nest 2 ("  " <> embraceN (chunksOf 1 (fmap plit pts))) <> ";"
  , ""
  ]
  where pts = envec . word64 . fromIntegral <$>  DVS.toList SM.transitionTableSimd
        envec t = Vec4
          (((t + 0x0c0c0c0c) .<. 32) .|. (t + 0x08080808))
          (((t + 0x04040404) .<. 32) .|.  t              )
          (((t + 0x0c0c0c0c) .<. 32) .|. (t + 0x08080808))
          (((t + 0x04040404) .<. 32) .|.  t              )

simdTransitionPhiTableSaturated256 :: Doc ()
simdTransitionPhiTableSaturated256 = vsep
  [ "__m256i simd_transition_phi_saturated_256[256] ="
  , nest 2 ("  " <> embraceN (chunksOf 1 (fmap plit pts))) <> ";"
  , ""
  ]
  where pts = zipWith envec (word64 . fromIntegral <$>  DVS.toList SM.phiTableSimd        )
                            (                           DVS.toList SM.transitionTableSimd )
        envec p t = Vec4
          (( p               .<. 32) .|.  p              )
          (( p               .<. 32) .|.  p              )
          (((t + 0x0c0c0c0c) .<. 32) .|. (t + 0x08080808))
          (((t + 0x04040404) .<. 32) .|.  t              )
