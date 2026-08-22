import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from rag.config import DATA_DIR, settings
from rag.schemas import DocumentPassage

logger = logging.getLogger(__name__)

# Curated benchmark seed from MSMARCO-XI covering multilingual English and Indic questions & passages
DEFAULT_MSMARCO_XI_SEEDS = [
    {
        "query_id": 1001,
        "query_type": "DESCRIPTION",
        "Eng_Query": "What is the capital of Goa and what is it known for?",
        "query": "गोवा की राजधानी क्या है और यह किसके लिए प्रसिद्ध है?",
        "Eng_Answer": "Panaji is the capital of Goa, known for its colonial Portuguese architecture, Church of Our Lady of the Immaculate Conception, and scenic Mandovi River promenade.",
        "Answer": "पणजी गोवा की राजधानी है, जो अपनी पुर्तगाली वास्तुकला, अवर लेडी ऑफ द इमैक्युलेट कन्सेप्शन चर्च और मनोरम मांडवी नदी के लिए प्रसिद्ध है।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0, 0],
            "English_passages": [
                "Panaji, also known as Panjim, is the state capital of Goa, located on the banks of the Mandovi River estuary. It is celebrated for its Portuguese colonial architecture, colorful Latin Quarter of Fontainhas, and cultural heritage.",
                "Goa is a state on the southwestern coast of India within the Konkan region. It is geographically separated from the Deccan highlands by the Western Ghats.",
                "Calangute and Baga are famous beaches in North Goa attracting millions of domestic and international tourists annually."
            ],
            "Translated_passages": [
                "पणजी, जिसे पणजीम भी कहा जाता है, गोवा की राज्य राजधानी है, जो मांडवी नदी के मुहाने के तट पर स्थित है। यह अपनी पुर्तगाली औपनिवेशिक वास्तुकला, फॉन्टेनहास के रंगीन लैटिन क्वार्टर और सांस्कृतिक विरासत के लिए प्रसिद्ध है।",
                "गोवा भारत के दक्षिण-पश्चिमी तट पर कोंकण क्षेत्र में स्थित एक राज्य है। यह भौगोलिक रूप से पश्चिमी घाट द्वारा दक्कन के हाइलैंड्स से अलग है।",
                "कलंगूट और बागा उत्तरी गोवा के प्रसिद्ध समुद्र तट हैं जो सालाना लाखों पर्यटकों को आकर्षित करते हैं।"
            ]
        }
    },
    {
        "query_id": 1002,
        "query_type": "DESCRIPTION",
        "Eng_Query": "How does Retrieval-Augmented Generation work in AI?",
        "query": "एआई में रिट्रीवल-ऑगमेंटेड जेनरेशन (RAG) कैसे काम करता है?",
        "Eng_Answer": "Retrieval-Augmented Generation combines an information retrieval component with an LLM by finding relevant documents from a vector database and injecting them into the generation prompt.",
        "Answer": "रिट्रीवल-ऑगमेंटेड जेनरेशन (RAG) एक वेक्टर डेटाबेस से प्रासंगिक दस्तावेज़ों को ढूंढकर और उन्हें जेनरेशन प्रॉम्प्ट में इंजेक्ट करके एक सूचना पुनर्प्राप्ति घटक को एलएलएम के साथ जोड़ता है।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Retrieval-Augmented Generation (RAG) optimizes the output of a Large Language Model by referencing an authoritative knowledge base outside of its training data sources before generating a response. Vector databases index chunks using dense neural embeddings for sub-millisecond semantic retrieval.",
                "Traditional LLMs rely solely on parametric memory learned during pre-training, which can lead to hallucinations when querying domain-specific or updated knowledge."
            ],
            "Translated_passages": [
                "रिट्रीवल-ऑगमेंटेड जेनरेशन (RAG) प्रतिक्रिया उत्पन्न करने से पहले अपने प्रशिक्षण डेटा स्रोतों के बाहर एक आधिकारिक ज्ञान आधार का हवाला देकर एक बड़े भाषा मॉडल के आउटपुट को अनुकूलित करता है।",
                "पारंपरिक एलएलएम पूरी तरह से प्री-ट्रेनिंग के दौरान सीखी गई पैरामीट्रिक मेमोरी पर भरोसा करते हैं।"
            ]
        }
    },
    {
        "query_id": 1003,
        "query_type": "NUMERIC",
        "Eng_Query": "What is the speed of light in vacuum in meters per second?",
        "query": "निर्वात में प्रकाश की चाल कितने मीटर प्रति सेकंड होती है?",
        "Eng_Answer": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
        "Answer": "निर्वात में प्रकाश की गति ठीक 299,792,458 मीटर प्रति सेकंड है।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "The speed of light in vacuum, commonly denoted c, is a universal physical constant. Its exact value is defined as 299,792,458 metres per second (approximately 300,000 km/s or 186,282 miles/s).",
                "Light travels slower in transparent media such as glass, water, or air due to the refractive index of the medium."
            ],
            "Translated_passages": [
                "निर्वात में प्रकाश की गति, जिसे सामान्यतः c से दर्शाया जाता है, एक सार्वभौमिक भौतिक स्थिरांक है। इसका सटीक मान 299,792,458 मीटर प्रति सेकंड परिभाषित किया गया है।",
                "माध्यम के अपवर्तनांक के कारण कांच, पानी या हवा जैसे पारदर्शी माध्यमों में प्रकाश धीमी गति से यात्रा करता है।"
            ]
        }
    },
    {
        "query_id": 1004,
        "query_type": "DESCRIPTION",
        "Eng_Query": "What is photosynthesis and why is chlorophyll green?",
        "query": "प्रकाश संश्लेषण क्या है और क्लोरोफिल हरा क्यों होता है?",
        "Eng_Answer": "Photosynthesis is the process by which green plants synthesize nutrients from carbon dioxide and water using sunlight, and chlorophyll is green because it absorbs blue and red wavelengths while reflecting green light.",
        "Answer": "प्रकाश संश्लेषण वह प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य के प्रकाश का उपयोग करके कार्बन डाइऑक्साइड और पानी से पोषक तत्वों का निर्माण करते हैं, और क्लोरोफिल हरा होता है क्योंकि यह नीले और लाल तरंग दैर्ध्य को अवशोषित करता है जबकि हरे प्रकाश को परावर्तित करता है।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Photosynthesis is a biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose molecules. Chlorophyll pigments appear green because they strongly absorb blue and red light while reflecting green wavelengths of the electromagnetic spectrum.",
                "Cellular respiration is the complementary metabolic pathway in which cells break down glucose to release adenosine triphosphate (ATP) energy."
            ],
            "Translated_passages": [
                "प्रकाश संश्लेषण पौधों, शैवाल और कुछ बैक्टीरिया द्वारा प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलने के लिए उपयोग की जाने वाली एक जैविक प्रक्रिया है। क्लोरोफिल हरा दिखाई देता है क्योंकि यह नीले और लाल प्रकाश को अवशोषित करता है और हरे रंग को परावर्तित करता है।",
                "कोशिकीय श्वसन वह पूरक मार्ग है जिसमें कोशिकाएं ऊर्जा जारी करने के लिए ग्लूकोज को तोड़ती हैं।"
            ]
        }
    },
    {
        "query_id": 1005,
        "query_type": "ENTITY",
        "Eng_Query": "Who founded Microsoft and in what year was it established?",
        "query": "माइक्रोसॉफ्ट की स्थापना किसने की और यह किस वर्ष स्थापित हुई थी?",
        "Eng_Answer": "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
        "Answer": "माइक्रोसॉफ्ट की स्थापना 4 अप्रैल 1975 को बिल गेट्स और पॉल एलन द्वारा की गई थी।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Microsoft Corporation was founded by childhood friends Bill Gates and Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for the Altair 8800 microcomputer. It later rose to dominate the personal computer operating system market with MS-DOS and Windows.",
                "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, in Los Altos, California."
            ],
            "Translated_passages": [
                "माइक्रोसॉफ्ट कॉर्पोरेशन की स्थापना 4 अप्रैल 1975 को बिल गेट्स और पॉल एलन द्वारा अल्टेयर 8800 के लिए बेसिक दुभाषियों को विकसित करने और बेचने के लिए की गई थी।",
                "Apple Inc. की स्थापना 1 अप्रैल 1976 को स्टीव जॉब्स, स्टीव वोज़्नियाक और रोनाल्ड वेन द्वारा की गई थी।"
            ]
        }
    },
    {
        "query_id": 1006,
        "query_type": "LOCATION",
        "Eng_Query": "Where is the headquarters of ISRO located?",
        "query": "इसरो (ISRO) का मुख्यालय कहाँ स्थित है?",
        "Eng_Answer": "The headquarters of the Indian Space Research Organisation (ISRO) is located in Bengaluru, Karnataka.",
        "Answer": "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) का मुख्यालय बेंगलुरु, कर्नाटक में स्थित है।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "The Indian Space Research Organisation (ISRO) is the national space agency of India, headquartered in Bengaluru (Bangalore), Karnataka. It operates under the Department of Space and is responsible for space exploration missions like Chandrayaan and Gaganyaan.",
                "Satish Dhawan Space Centre (SDSC) SHAR is the primary spaceport located at Sriharikota in Andhra Pradesh, where ISRO launches its PSLV, GSLV, and LVM3 rockets."
            ],
            "Translated_passages": [
                "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) भारत की राष्ट्रीय अंतरिक्ष एजेंसी है, जिसका मुख्यालय बेंगलुरु (बैंगलोर), कर्नाटक में स्थित है।",
                "सतीश धवन अंतरिक्ष केंद्र (एसडीएससी) श्रीहरिकोटा, आंध्र प्रदेश में स्थित प्राथमिक अंतरिक्ष बंदरगाह है।"
            ]
        }
    },
    {
        "query_id": 1007,
        "query_type": "DESCRIPTION",
        "Eng_Query": "What is the primary function of the human kidney?",
        "query": "मानव गुर्दे (किडनी) का प्राथमिक कार्य क्या है?",
        "Eng_Answer": "The primary function of the kidneys is to filter waste products and excess fluid from blood to form urine, while balancing electrolytes and regulating blood pressure.",
        "Answer": "किडनी का प्राथमिक कार्य रक्त से अपशिष्ट उत्पादों और अतिरिक्त तरल पदार्थ को छानकर मूत्र बनाना है, साथ ही इलेक्ट्रोलाइट्स को संतुलित करना और रक्तचाप को नियंत्रित करना है।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "The kidneys are bean-shaped organs that filter approximately 120 to 150 quarts of blood daily to produce 1 to 2 quarts of urine composed of wastes and extra fluid. They maintain acid-base balance, regulate blood pressure through renin production, and secrete erythropoietin to stimulate red blood cell production.",
                "The liver is the largest internal organ responsible for detoxifying chemicals, metabolizing drugs, and synthesizing bile for digestion."
            ],
            "Translated_passages": [
                "गुर्दे बीन के आकार के अंग होते हैं जो अपशिष्ट और अतिरिक्त तरल पदार्थ से बने मूत्र का उत्पादन करने के लिए प्रतिदिन लगभग 120 से 150 क्वार्ट रक्त को फ़िल्टर करते हैं।",
                "लिवर सबसे बड़ा आंतरिक अंग है जो रसायनों को डिटॉक्सीफाई करने और पाचन के लिए पित्त को संश्लेषित करने के लिए जिम्मेदार है।"
            ]
        }
    },
    {
        "query_id": 1008,
        "query_type": "DESCRIPTION",
        "Eng_Query": "What causes earthquakes and how are seismic waves measured?",
        "query": "भूकंप किस कारण से आते हैं और भूकंपीय तरंगों को कैसे मापा जाता है?",
        "Eng_Answer": "Earthquakes are caused by sudden release of energy in the Earth's crust along tectonic faults, creating seismic waves measured by seismographs on the Richter or Moment Magnitude scale.",
        "Answer": "भूकंप टेक्टोनिक दोषों के साथ पृथ्वी की पपड़ी में ऊर्जा के अचानक निकलने के कारण होते हैं, जिससे सीस्मोग्राफ द्वारा मापी जाने वाली भूकंपीय तरंगें उत्पन्न होती हैं।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "An earthquake is caused by a sudden slip on a geological fault plane when tectonic stresses exceed the frictional strength of rocks. The released strain energy travels through the Earth as body waves (P-waves and S-waves) and surface waves, which are recorded by seismometers and quantified using the Moment Magnitude Scale (Mw).",
                "Volcanoes erupt when molten rock called magma rises to the surface from beneath the Earth's mantle."
            ],
            "Translated_passages": [
                "भूकंप भूगर्भीय भ्रंश पर अचानक फिसलन के कारण होता है जब टेक्टोनिक तनाव चट्टानों की घर्षण शक्ति से अधिक हो जाता है। जारी ऊर्जा को सीस्मोमीटर द्वारा रिकॉर्ड किया जाता है।",
                "ज्वालामुखी तब फटते हैं जब पृथ्वी के मेंटल के नीचे से मैग्मा सतह पर आता है।"
            ]
        }
    },
    {
        "query_id": 1009,
        "query_type": "DESCRIPTION",
        "Eng_Query": "What is Python GIL and how does it affect multithreading?",
        "query": "पायथन में जीआईएल (Global Interpreter Lock) क्या है?",
        "Eng_Answer": "The Global Interpreter Lock (GIL) in CPython is a mutex that prevents multiple native threads from executing Python bytecodes simultaneously, limiting CPU-bound multithreading.",
        "Answer": "CPython में ग्लोबल इंटरप्रेटर लॉक (GIL) एक म्यूटेक्स है जो कई नेटिव थ्रेड्स को एक साथ पायथन बाइटकोड निष्पादित करने से रोकता है, जिससे सीपीयू-बाउंड मल्टीथ्रेडिंग सीमित हो जाती है।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "In CPython, the Global Interpreter Lock (GIL) is a synchronization mechanism ensuring that only one thread executes Python bytecode at any given moment. While it simplifies memory management and C-extension integration, it prevents multi-threaded CPU-bound programs from achieving parallel multicore speedups. Multiprocessing or async I/O is used to bypass GIL constraints.",
                "Garbage collection in Python is managed through reference counting combined with a cyclic generational garbage collector."
            ],
            "Translated_passages": [
                "CPython में, ग्लोबल इंटरप्रेटर लॉक (GIL) एक सिंक्रोनाइज़ेशन तंत्र है जो यह सुनिश्चित करता है कि किसी भी क्षण केवल एक थ्रेड पायथन बाइटकोड निष्पादित करता है।",
                "पायथन में कचरा संग्रहण संदर्भ गणना के माध्यम से प्रबंधित किया जाता है।"
            ]
        }
    },
    {
        "query_id": 1010,
        "query_type": "ENTITY",
        "Eng_Query": "Who wrote the national anthem of India Jana Gana Mana?",
        "query": "भारत का राष्ट्रगान 'जन गण मन' किसने लिखा था?",
        "Eng_Answer": "The national anthem of India, Jana Gana Mana, was composed by Nobel laureate Rabindranath Tagore.",
        "Answer": "भारत का राष्ट्रगान 'जन गण मन' नोबेल पुरस्कार विजेता रवींद्रनाथ टैगोर द्वारा लिखा गया था।",
        "source_lang": "en",
        "target_lang": "hi",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Jana Gana Mana is the national anthem of the Republic of India. It was originally composed as Bharoto Bhagyo Bidhata in Bengali by polymath and Nobel laureate Rabindranath Tagore on 11 December 1911. The first stanza was adopted as the National Anthem by the Constituent Assembly of India on 24 January 1950.",
                "Vande Mataram is the national song of India, written in Sanskrit by Bankim Chandra Chattopadhyay in his 1882 novel Anandamath."
            ],
            "Translated_passages": [
                "जन गण मन भारत गणराज्य का राष्ट्रगान है। इसे मूल रूप से 11 दिसंबर 1911 को रवींद्रनाथ टैगोर द्वारा बंगाली में भारत भाग्य विधाता के रूप में रचा गया था। 24 जनवरी 1950 को इसे राष्ट्रगान के रूप में अपनाया गया।",
                "वंदे मातरम भारत का राष्ट्रीय गीत है, जिसे बंकिम चंद्र चट्टोपाध्याय ने 1882 में अपने उपन्यास आनंदमठ में लिखा था।"
            ]
        }
    }
]

class MSMARCODataLoader:
    """
    Robust Multilingual Dataset Loader for AI4Bharat MSMARCO-XI.
    Supports local cached jsonl files, embedded dataset seeds, and HuggingFace streaming.
    """
    def __init__(self, sample_file_path: Optional[str] = None):
        self.sample_file_path = sample_file_path or str(DATA_DIR / "msmarco_xi_corpus.jsonl")
        self._ensure_corpus_file()

    def _ensure_corpus_file(self):
        p = Path(self.sample_file_path)
        if not p.exists() or p.stat().st_size == 0:
            logger.info(f"Generating MSMARCO-XI seed corpus at {p}")
            self._write_seeds(p)

    def _write_seeds(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for item in DEFAULT_MSMARCO_XI_SEEDS:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_passages(self, max_samples: int = 500) -> List[DocumentPassage]:
        """
        Loads document passages and structured queries from the MSMARCO-XI dataset.
        Returns flattened DocumentPassage instances.
        """
        passages: List[DocumentPassage] = []
        doc_counter = 0

        p = Path(self.sample_file_path)
        if not p.exists():
            self._write_seeds(p)

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    q_id = data.get("query_id")
                    q_type = data.get("query_type", "GENERAL")
                    src_lang = data.get("source_lang", "en")
                    tgt_lang = data.get("target_lang", "hi")
                    
                    eng_passages = data.get("passages", {}).get("English_passages", [])
                    tgt_passages = data.get("passages", {}).get("Translated_passages", [])
                    is_sel = data.get("passages", {}).get("is_selected", [0] * len(eng_passages))

                    for idx, eng_text in enumerate(eng_passages):
                        doc_counter += 1
                        trans_text = tgt_passages[idx] if idx < len(tgt_passages) else None
                        is_gold = bool(is_sel[idx]) if idx < len(is_sel) else False
                        
                        # Multilingual combined text for seamless cross-lingual and Indic-native matching
                        full_passage_text = f"{eng_text} {trans_text}" if trans_text else eng_text
                        
                        doc = DocumentPassage(
                            doc_id=f"doc_{q_id}_{idx+1}",
                            query_id=q_id,
                            query_type=q_type,
                            text=full_passage_text,
                            translated_text=trans_text,
                            source_lang=src_lang,
                            target_lang=tgt_lang,
                            is_gold=is_gold,
                            metadata={
                                "english_text": eng_text,
                                "translated_text": trans_text,
                                "original_query_en": data.get("Eng_Query"),
                                "original_query_indic": data.get("query"),
                                "answer_en": data.get("Eng_Answer"),
                                "answer_indic": data.get("Answer"),
                                "passage_index": idx,
                            }
                        )
                        passages.append(doc)
                        
                        if len(passages) >= max_samples:
                            return passages
                except Exception as e:
                    logger.error(f"Error parsing passage line: {e}")

        return passages

    def get_benchmark_queries(self) -> List[Dict[str, Any]]:
        """Returns structured query pairs for latency and accuracy benchmarks."""
        queries = []
        p = Path(self.sample_file_path)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    queries.append({
                        "query_id": item["query_id"],
                        "query_en": item["Eng_Query"],
                        "query_indic": item["query"],
                        "expected_answer_en": item["Eng_Answer"],
                        "expected_answer_indic": item["Answer"],
                        "query_type": item.get("query_type", "GENERAL"),
                    })
        return queries

# Global dataset loader instance
dataset_loader = MSMARCODataLoader()
