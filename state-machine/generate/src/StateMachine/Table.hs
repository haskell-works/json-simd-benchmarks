{-# LANGUAGE BinaryLiterals #-}

module StateMachine.Table
  ( phiTable
  , phiTableSimd
  , transitionTable
  , transitionTableSimd
  ) where

import Data.Word
import HaskellWorks.Data.Bits.BitWise

import qualified Data.Vector               as DV
import qualified Data.Vector.Storable      as DVS
import qualified StateMachine.StateMachine as SM

data State = InJson | InString | InEscape | InValue deriving (Eq, Enum, Bounded, Show)

phiTable :: DV.Vector (DVS.Vector Word8)
phiTable = DV.generate 4 gos
  where gos :: Int -> DVS.Vector Word8
        gos vj = DVS.generate 256 go
          where vi = fromIntegral vj
                go :: Int -> Word8
                go uj = fromIntegral (snd (SM.stateMachine ui (toEnum vi)))
                  where ui = fromIntegral uj
{-# NOINLINE phiTable #-}

phiTableSimd :: DVS.Vector Word32
phiTableSimd = DVS.generate 256 go
  where go :: Int -> Word32
        go vj = (snd (SM.stateMachine vi SM.InJson  ) .<.  0) .|.
                (snd (SM.stateMachine vi SM.InString) .<.  8) .|.
                (snd (SM.stateMachine vi SM.InEscape) .<. 16) .|.
                (snd (SM.stateMachine vi SM.InValue ) .<. 24)
          where vi = fromIntegral vj
{-# NOINLINE phiTableSimd #-}

transitionTable :: DV.Vector (DVS.Vector Word8)
transitionTable = DV.generate 4 gos
  where gos :: Int -> DVS.Vector Word8
        gos vj = DVS.generate 256 go
          where vi = fromIntegral vj
                go :: Int -> Word8
                go uj = fromIntegral (fromEnum (fst (SM.stateMachine ui (toEnum vi))))
                  where ui = fromIntegral uj
{-# NOINLINE transitionTable #-}

transitionTableSimd :: DVS.Vector Word64
transitionTableSimd = DVS.generate 256 go
  where go :: Int -> Word64
        go vj = (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InJson  ))) .<.  0) .|.
                (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InString))) .<.  8) .|.
                (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InEscape))) .<. 16) .|.
                (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InValue ))) .<. 24)
          where vi = fromIntegral vj
{-# NOINLINE transitionTableSimd #-}
