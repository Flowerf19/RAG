import spacy
_nlp_cache = {}
def get_nlp(lang='vi'):
	"""
	Lấy spaCy model cho ngôn ngữ. lang='vi' (mặc định) hoặc 'en'.
	"""
	global _nlp_cache
	if lang not in _nlp_cache:
		if lang == 'en':
			_nlp_cache[lang] = spacy.load('en_core_web_sm')
		else:
			_nlp_cache[lang] = spacy.load('vi_core_news_sm')
	return _nlp_cache[lang]

def sent_tokenize(text, lang='vi'):
	"""
	Tách câu với spaCy. lang='vi' hoặc 'en'.
	"""
	nlp = get_nlp(lang)
	doc = nlp(text)
	return [sent.text for sent in doc.sents]
