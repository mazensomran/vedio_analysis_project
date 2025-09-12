from transformers import MarianMTModel, MarianTokenizer
import logging

logger = logging.getLogger(__name__)

class MarianTranslator:
    def __init__(self):
        try:
            model_name = 'Helsinki-NLP/opus-mt-en-ar'
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            logger.info(f"✅ تم تحميل نموذج الترجمة {model_name} بنجاح.")
        except Exception as e:
            self.tokenizer = None
            self.model = None
            logger.error(f"❌ فشل تحميل نموذج الترجمة {model_name}: {e}")

    def translate(self, text: str) -> str:
        if not text or self.model is None:
            return text # Return original text if model not loaded or text is empty
        try:
            batch = self.tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
            translated = self.model.generate(**batch)
            tgt_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            return tgt_text
        except Exception as e:
            logger.error(f"❌ خطأ في الترجمة: {e}")
            return text
