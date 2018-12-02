{-# LANGUAGE BinaryLiterals #-}

module StateMachine.StateMachine
  ( State(..)
  , stateMachine
  ) where

import Data.Word

import qualified StateMachine.Word8 as W8

data State = InJson | InString | InEscape | InValue deriving (Eq, Enum, Bounded, Show)

stateMachine :: Word8 -> State -> (State, Word32)
stateMachine c InJson   | W8.isOpen c         = (InJson  , 0b110)
stateMachine c InJson   | W8.isClose c        = (InJson  , 0b001)
stateMachine c InJson   | W8.isDelim c        = (InJson  , 0b000)
stateMachine c InJson   | W8.isValueChar c    = (InValue , 0b111)
stateMachine c InJson   | W8.isDoubleQuote c  = (InString, 0b111)
stateMachine _ InJson   | otherwise           = (InJson  , 0b000)
stateMachine c InString | W8.isDoubleQuote c  = (InJson  , 0b000)
stateMachine c InString | W8.isBackSlash c    = (InEscape, 0b000)
stateMachine _ InString | otherwise           = (InString, 0b000)
stateMachine _ InEscape | otherwise           = (InString, 0b000)
stateMachine c InValue  | W8.isOpen c         = (InJson  , 0b110)
stateMachine c InValue  | W8.isClose c        = (InJson  , 0b001)
stateMachine c InValue  | W8.isDelim c        = (InJson  , 0b000)
stateMachine c InValue  | W8.isValueChar c    = (InValue , 0b000)
stateMachine _ InValue  | otherwise           = (InJson  , 0b000)
{-# INLINE stateMachine #-}
