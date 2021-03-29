
"""Python implementation of ESS (Event Sequence Similarity, a modified BLEU) and smooth-ESS.
This module provides a Python implementation of ESS and smooth-ESS.
Smooth ESS is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""
import os
import collections
import math
import json
import copy

TOKEN_SPEARTOR = None # '\t'

def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    max_order = min(reference_length, translation_length, max_order)
    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)


def ess_bleu_eval_func(ref_file, trans_file, max_gram=5, smooth=False):
  """Compute BLEU scores"""
  #max_gram = 4
  #smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with open(reference_filename, 'r', encoding='utf-8') as rfh:
      reference_text.append(rfh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      if not reference.strip('\n'):
        continue
      reference_list.append(reference.split(TOKEN_SPEARTOR))
    if reference_list:
      per_segment_references.append(reference_list)

  translations = []
  with open(trans_file, "r", encoding='utf-8') as rfh:
    for line in rfh:
      if not line.strip('\n'):
        continue      
      translations.append(line.split(TOKEN_SPEARTOR))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = compute_bleu(
      per_segment_references, translations, max_gram, smooth)
  return 100 * bleu_score

def self_ess_bleu_eval_func( trans_file, max_gram=5, smooth=False):
  """Compute BLEU scores"""
  #max_gram = 4
  #smooth = False

  ref_files = [trans_file]
  reference_text = []
  for reference_filename in ref_files:
    with open(reference_filename, 'r', encoding='utf-8') as rfh:
      reference_text.append(rfh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      if not reference.strip('\n'):
        continue
      reference_list.append(reference.split(TOKEN_SPEARTOR))
    if reference_list:
      per_segment_references.append(reference_list)

  translations = []
  with open(trans_file, "r", encoding='utf-8') as rfh:
    for line in rfh:
      translations.append(line.split(TOKEN_SPEARTOR))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = compute_bleu(
      per_segment_references, translations, max_gram, smooth)
  return 100 * bleu_score