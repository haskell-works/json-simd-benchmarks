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
phiTable = DV.constructN 4 gos
  where gos :: DV.Vector (DVS.Vector Word8) -> DVS.Vector Word8
        gos v = DVS.constructN 256 go
          where vi = DV.length v
                go :: DVS.Vector Word8 -> Word8
                go u = fromIntegral (snd (SM.stateMachine ui (toEnum vi)))
                  where ui = fromIntegral (DVS.length u)
{-# NOINLINE phiTable #-}

phiTableSimd :: DVS.Vector Word32
phiTableSimd = DVS.constructN 256 go
  where go :: DVS.Vector Word32 -> Word32
        go v =  (snd (SM.stateMachine vi SM.InJson  ) .<.  0) .|.
                (snd (SM.stateMachine vi SM.InString) .<.  8) .|.
                (snd (SM.stateMachine vi SM.InEscape) .<. 16) .|.
                (snd (SM.stateMachine vi SM.InValue ) .<. 24)
          where vi = fromIntegral (DVS.length v)
{-# NOINLINE phiTableSimd #-}

transitionTable :: DV.Vector (DVS.Vector Word8)
transitionTable = DV.constructN 4 gos
  where gos :: DV.Vector (DVS.Vector Word8) -> DVS.Vector Word8
        gos v = DVS.constructN 256 go
          where vi = DV.length v
                go :: DVS.Vector Word8 -> Word8
                go u = fromIntegral (fromEnum (fst (SM.stateMachine ui (toEnum vi))))
                  where ui = fromIntegral (DVS.length u)
{-# NOINLINE transitionTable #-}

transitionTableSimd :: DVS.Vector Word64
transitionTableSimd = DVS.constructN 256 go
  where go :: DVS.Vector Word64 -> Word64
        go v =  (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InJson  ))) .<.  0) .|.
                (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InString))) .<.  8) .|.
                (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InEscape))) .<. 16) .|.
                (fromIntegral (fromEnum (fst (SM.stateMachine vi SM.InValue ))) .<. 24)
          where vi = fromIntegral (DVS.length v)
{-# NOINLINE transitionTableSimd #-}
