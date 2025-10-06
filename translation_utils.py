import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
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


class M2M100Translator:
    def __init__(self):
        model_name = "facebook/m2m100_418M"
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ تم تحميل نموذج {model_name} بنجاح على الجهاز {self.device}.")

    def translate(self, text: str) -> str:
      if not text:
          return text
      try:
        self.tokenizer.src_lang = "en"
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=100
        ).to(self.device)
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.get_lang_id("ar"),
                max_length=150,
                min_length=30,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2
            )
        tgt_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return tgt_text
      except Exception as e:
        print(f"❌ خطأ في الترجمة: {e}")
        return text
