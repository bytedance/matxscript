// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of GeneratorIterator is inspired by pythran.
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>

#include <matxscript/runtime/exceptions/exceptions.h>

namespace matxscript {
namespace runtime {

template <class T>
struct GeneratorIterator : std::iterator<std::forward_iterator_tag,
                                         typename T::result_type,
                                         ptrdiff_t,
                                         typename T::result_type*,
                                         typename T::result_type /* no ref */> {
  T the_generator;
  GeneratorIterator();
  GeneratorIterator(T const& a_generator);
  GeneratorIterator& operator++();
  typename T::result_type operator*() const;
  bool operator!=(GeneratorIterator<T> const& other) const;
  bool operator==(GeneratorIterator<T> const& other) const;
  bool operator<(GeneratorIterator<T> const& other) const;
};

template <class T>
GeneratorIterator<T>::GeneratorIterator() : the_generator() {
  the_generator.SetState(-1);
}  // this represents the end

template <class T>
GeneratorIterator<T>::GeneratorIterator(T const& a_generator) : the_generator(a_generator) {
}

template <class T>
GeneratorIterator<T>& GeneratorIterator<T>::operator++() {
  try {
    the_generator.next();
  } catch (StopIteration const&) {
    the_generator.SetState(-1);
  }
  return *this;
}

template <class T>
typename T::result_type GeneratorIterator<T>::operator*() const {
  return *the_generator;
}

template <class T>
bool GeneratorIterator<T>::operator!=(GeneratorIterator<T> const& other) const {
  assert(other.the_generator.GetState() == -1 || the_generator.GetState() == -1);
  return the_generator.GetState() != other.the_generator.GetState();
}

template <class T>
bool GeneratorIterator<T>::operator==(GeneratorIterator<T> const& other) const {
  assert(other.the_generator.GetState() == -1 || the_generator.GetState() == -1);
  return the_generator.GetState() == other.the_generator.GetState();
}

template <class T>
bool GeneratorIterator<T>::operator<(GeneratorIterator<T> const& other) const {
  assert(other.the_generator.GetState() == -1 || the_generator.GetState() == -1);
  return the_generator.GetState() != other.the_generator.GetState();
}

}  // namespace runtime
}  // namespace matxscript
