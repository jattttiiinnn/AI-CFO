# import json
# import os
# import re
# from io import BytesIO
# from datetime import datetime
# import google.generativeai as genai
# import pandas as pd
# import pdfplumber
# from reportlab.lib.pagesizes import A4
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# from django.conf import settings
# from django.http import HttpResponse, JsonResponse
# from django.shortcuts import render, redirect
# from django.urls import reverse
# from django.views.decorators.csrf import csrf_exempt
# from .forms import UploadForm
# from PyPDF2 import PdfReader


# # ---------- Configuration ----------
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# METRIC_SYNONYMS = {
#     'revenue': ['revenue', 'sales', 'turnover', 'total revenue'],
#     'expenses': ['expense', 'expenses', 'operating expenses', 'opex', 'total expenses', 'costs'],
#     'cash_reserves': ['cash', 'cash reserves', 'cash balance', 'cash and cash equivalents'],
#     'assets': ['assets', 'total assets'],
#     'liabilities': ['liabilities', 'total liabilities'],
#     'equity': ['equity', 'shareholder equity', "shareholders' equity", 'net assets'],
#     'debt': ['debt', 'loans', 'borrowings', 'total debt'],
# }

# ZOMATO_SECTIONS = {
#     'foodDelivery': ['food delivery', 'delivery'],
#     'quickCommerce': ['quick commerce', 'blinkit'],
#     'goingOut': ['going-out', 'going out', 'dining out', 'dining'],
#     'b2bSupplies': ['b2b supplies', 'hyperpure']
# }


# # ---------- Helper Functions ----------

# def call_ai_model(prompt, model="gemini-1.5-flash"):
#     """
#     Calls Gemini Pro with a strict JSON prompt and validates the output.
#     Always returns a Python dict or error dict.
#     """
#     try:
#         model_obj = genai.GenerativeModel(model)
#         response = model_obj.generate_content(prompt)
#         text = response.text.strip() if hasattr(response, "text") else str(response)
#         # Clean markdown JSON blocks if present
#         if text.startswith("```"):
#             text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
#         # Try parsing JSON
#         try:
#             return json.loads(text)
#         except Exception:
#             return {"error": "AI response was not valid JSON", "raw": text}
#     except Exception as e:
#         return {"error": f"AI call failed: {e}"}


# def normalize_col(name: str) -> str:
#     """Normalize column name for comparison."""
#     return str(name).strip().lower()


# def _column_match_score(col_name: str, synonyms: list[str]) -> float:
#     """Compute a simple fuzzy score [0..1] based on case-insensitive substring and similarity."""
#     from difflib import SequenceMatcher
#     n = normalize_col(col_name)
#     best = 0.0
    
#     for syn in synonyms:
#         syn_norm = syn.lower()
#         # Exact match
#         if syn_norm == n:
#             return 1.0
#         # Substring match
#         if syn_norm in n or n in syn_norm:
#             best = max(best, 0.8)
#         # Similarity ratio
#         ratio = SequenceMatcher(None, n, syn_norm).ratio()
#         best = max(best, ratio)
    
#     return best


# def _to_number(s) -> float | None:
#     """Convert string to number, handling currency symbols and formatting."""
#     if s is None or s == '':
#         return None
    
#     s = str(s).strip()
#     if not s:
#         return None
    
#     # Check for negative indicators
#     neg = s.startswith('-') or s.startswith('(')
    
#     # Remove common non-numeric characters
#     s = re.sub(r'[^\d,.\-+()]', '', s)
#     s = s.replace('(', '').replace(')', '')
    
#     # Handle comma/dot formatting
#     if ',' in s and '.' in s:
#         s = s.replace(',', '')
#     elif s.count(',') >= 1 and s.count('.') == 0:
#         s = s.replace(',', '')
    
#     try:
#         num = float(s)
#         return -num if neg else num
#     except Exception:
#         return None


# def _relevant_pdf_pages(pdf, keywords: list[str], max_pages: int = 50) -> list[int]:
#     """Find pages containing relevant keywords."""
#     relevant_pages = []
    
#     for i, page in enumerate(pdf.pages[:max_pages]):
#         try:
#             text = page.extract_text() or ""
#             text_lower = text.lower()
            
#             # Check if any keyword appears in the page
#             if any(kw.lower() in text_lower for kw in keywords):
#                 relevant_pages.append(i)
#         except Exception:
#             continue
    
#     return relevant_pages


# def safe_float(val):
#     """Safely convert value to float."""
#     try:
#         return float(val)
#     except (ValueError, TypeError):
#         return 0.0


# def extract_text_from_pdf(file):
#     """
#     Extracts text from a PDF file.
#     Accepts either a file path (str) or bytes.
#     Returns extracted text or error string.
#     """
#     text = ""
#     try:
#         if isinstance(file, str):
#             # File path
#             reader = PdfReader(file)
#         elif isinstance(file, bytes):
#             # Bytes - wrap in BytesIO to make it file-like
#             from io import BytesIO
#             reader = PdfReader(BytesIO(file))
#         else:
#             return "⚠️ Invalid file input."
        
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#     except Exception as e:
#         return f"⚠️ Error reading PDF: {e}"
#     return text


# def map_columns_to_metrics(df):
#     """Map DataFrame columns to financial metrics."""
#     mapped = {k: None for k in METRIC_SYNONYMS.keys()}
    
#     for metric, synonyms in METRIC_SYNONYMS.items():
#         best_col = None
#         best_score = 0.0
        
#         for col in df.columns:
#             score = _column_match_score(str(col), synonyms)
#             if score > best_score and score >= 0.6:
#                 best_score = score
#                 best_col = col
        
#         if best_col is not None:
#             # Get the first non-null numeric value from this column
#             series = pd.to_numeric(df[best_col], errors='coerce')
#             first_valid = series.dropna().iloc[0] if not series.dropna().empty else None
#             mapped[metric] = first_valid
    
#     return mapped


# def extract_metrics_from_csv_chunked(file_path: str, chunk_size: int = 1000) -> dict:
#     """Extract metrics from CSV using chunked processing for large files."""
#     mapped = {k: None for k in METRIC_SYNONYMS.keys()}
    
#     try:
#         # Read first chunk to get column structure
#         first_chunk = pd.read_csv(file_path, nrows=chunk_size)
#         return map_columns_to_metrics(first_chunk)
#     except Exception as e:
#         print(f"Error reading CSV: {e}")
#         return mapped


# def extract_metrics_from_pdf(file_path: str) -> dict:
#     """Extract metrics from a PDF. Try tables on relevant pages first; fallback to text scanning."""
#     metrics = {k: None for k in METRIC_SYNONYMS.keys()}
    
#     try:
#         with pdfplumber.open(file_path) as pdf:
#             # Build keyword list from all metric synonyms
#             keywords = []
#             for v in METRIC_SYNONYMS.values():
#                 keywords.extend(v)
            
#             pages_idx = _relevant_pdf_pages(pdf, keywords)[:20]
            
#             # Try to extract from tables first
#             for i in pages_idx:
#                 page = pdf.pages[i]
#                 try:
#                     tables = page.extract_tables() or []
#                 except Exception:
#                     tables = []
                
#                 for tbl in tables:
#                     if not tbl or len(tbl) < 2 or len(tbl[0]) < 2:
#                         continue
                    
#                     header = [h if h is not None else '' for h in tbl[0]]
#                     rows = [[c if c is not None else '' for c in r] for r in tbl[1:]]
                    
#                     # Limit excessive rows for performance
#                     if len(rows) > 1000:
#                         rows = rows[:1000]
                    
#                     try:
#                         df = pd.DataFrame(rows, columns=header)
#                         extracted = map_columns_to_metrics(df)
                        
#                         # Update metrics with non-null values
#                         for key, value in extracted.items():
#                             if value is not None and metrics[key] is None:
#                                 metrics[key] = value
#                     except Exception:
#                         continue
            
#             # Fallback to text extraction if no metrics found
#             if all(v is None for v in metrics.values()):
#                 full_text = ""
#                 for i in pages_idx:
#                     page = pdf.pages[i]
#                     text = page.extract_text() or ""
#                     full_text += text
                
#                 # Text pattern matching
#                 for key, synonyms in METRIC_SYNONYMS.items():
#                     if metrics[key] is not None:
#                         continue
                    
#                     for syn in synonyms:
#                         syn_esc = re.escape(syn)
#                         pattern = rf"(?i)\b{syn_esc}\b\s*[:\-]?\s*\(?[-+]?\$?[\d,.]+\)?"
#                         for m in re.finditer(pattern, full_text):
#                             num_match = re.search(r"\(?[-+]?\$?[\d,.]+\)?$", m.group(0))
#                             if num_match:
#                                 val = _to_number(num_match.group(0))
#                                 if val is not None:
#                                     metrics[key] = val
#                                     break
#                         if metrics[key] is not None:
#                             break
    
#     except Exception as e:
#         print(f"Error extracting from PDF: {e}")
    
#     return metrics


# def _find_section_key(text: str, section_map: dict) -> str | None:
#     """Find which section a text belongs to."""
#     t = text.lower()
#     for key, synonyms in section_map.items():
#         for s in synonyms:
#             if s in t:
#                 return key
#     return None


# # ---------- Main Analysis Function ----------

# @csrf_exempt
# def analyze_financials(request):
#     """
#     Analyze financials from POSTed JSON or dict.
#     Returns JsonResponse or dict.
#     """
#     if isinstance(request, dict):
#         data = request
#     else:
#         if request.method != "POST":
#             return JsonResponse({"error": "Only POST allowed"}, status=405)
#         try:
#             data = json.loads(request.body)
#         except Exception:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)

#     # Extract and validate input
#     company = data.get("company", "Unknown Company")
#     period = data.get("period", "Not specified")
#     revenue = safe_float(data.get("revenue", 0))
#     expenses = safe_float(data.get("expenses", 0))
#     net_income = revenue - expenses if revenue and expenses else None
#     burn_rate = safe_float(data.get("monthly_burn_rate", 0))
#     debt = safe_float(data.get("debt", 0))
#     equity = safe_float(data.get("equity", 0))

#     metrics = {
#         "Revenue": revenue,
#         "Expenses": expenses,
#         "Net Income": net_income if net_income is not None else "Data not provided",
#         "Monthly Burn Rate": burn_rate if burn_rate else "Data not provided",
#         "Runway (Months)": (
#             round(revenue / burn_rate, 1) if burn_rate and burn_rate > 0 and revenue > 0 else "Not applicable (profitable or no burn)"
#         ),
#         "Profit Margin (%)": (
#             round((net_income / revenue) * 100, 2) if revenue > 0 and net_income is not None else "Data not provided"
#         ),
#         "Debt-to-Equity Ratio": (
#             round(debt / equity, 2) if equity > 0 else "Data not provided"
#         ),
#     }

#     # Risks detection
#     risks = []
#     if isinstance(metrics["Profit Margin (%)"], (int, float)) and metrics["Profit Margin (%)"] < 5:
#         risks.append({
#             "level": "High",
#             "title": "Low Profitability",
#             "description": "Profit margins are below 5%, indicating potential cost or pricing challenges."
#         })
#     if isinstance(metrics["Runway (Months)"], (int, float)) and metrics["Runway (Months)"] < 6:
#         risks.append({
#             "level": "High",
#             "title": "Short Cash Runway",
#             "description": "Cash reserves may not sustain operations beyond 6 months at current burn rate."
#         })
#     if isinstance(metrics["Debt-to-Equity Ratio"], (int, float)) and metrics["Debt-to-Equity Ratio"] > 2:
#         risks.append({
#             "level": "High",
#             "title": "High Financial Leverage",
#             "description": "Debt-to-Equity ratio above 2 indicates excessive reliance on debt."
#         })
#     if revenue > 0 and expenses > revenue:
#         risks.append({
#             "level": "Medium",
#             "title": "Expenses Exceed Revenue",
#             "description": "Operating expenses are higher than revenue, leading to negative net income."
#         })
#     if not risks:
#         risks.append({
#             "level": "Low",
#             "title": "No critical risks detected",
#             "description": "Financials appear stable, but continuous monitoring is advised."
#         })

#     recommendations = [
#         {
#             "priority": 1,
#             "action": "Improve liquidity management",
#             "details": "Maintain at least 9–12 months runway for financial safety."
#         },
#         {
#             "priority": 2,
#             "action": "Optimize cost structure",
#             "details": "Review operating expenses and reduce non-essential spend to protect margins."
#         },
#         {
#             "priority": 3,
#             "action": "Balance capital structure",
#             "details": "If debt levels are high, explore equity infusion or refinancing."
#         }
#     ]

#     missing_data = []
#     if not debt:
#         missing_data.append("Debt details")
#     if not equity:
#         missing_data.append("Equity details")
#     if not burn_rate:
#         missing_data.append("Cash flow / burn rate")
#     if not expenses:
#         missing_data.append("Operating expenses breakdown")

#     exec_summary = f"""
#     For {company} during {period}, reported revenue was {revenue:,.0f} with expenses of {expenses:,.0f}.
#     Net income stands at {net_income:,.0f} ({metrics['Profit Margin (%)']}% margin).
#     Current runway is {metrics['Runway (Months)']} months and Debt-to-Equity is {metrics['Debt-to-Equity Ratio']}.
#     Key focus areas: cost discipline, liquidity management, and capital structure optimization.
#     """

#     response = {
#         "company": company,
#         "period": period,
#         "executive_summary": exec_summary.strip(),
#         "metrics": metrics,
#         "risks": risks,
#         "recommendations": recommendations,
#         "missing_data": missing_data
#     }

#     if not isinstance(request, dict):
#         return JsonResponse(response, safe=False, status=200)
#     return response


# def parse_zomato_pdf(file_path: str) -> dict:
#     """Extract structured data for Zomato annual report."""
#     segments = {
#         'foodDelivery': {},
#         'quickCommerce': {},
#         'goingOut': {},
#         'b2bSupplies': {},
#     }
#     esg = {}

#     try:
#         with pdfplumber.open(file_path) as pdf:
#             kw = ['food delivery', 'quick commerce', 'going-out', 'going out', 'hyperpure', 'b2b', 'gov', 'revenue', 'yoy']
#             pages_idx = _relevant_pdf_pages(pdf, kw)[:20]
            
#             for i in pages_idx:
#                 page = pdf.pages[i]
#                 tables = []
#                 try:
#                     tables = page.extract_tables() or []
#                 except Exception:
#                     pass
                
#                 for tbl in tables:
#                     if not tbl or len(tbl) < 2 or len(tbl[0]) < 2:
#                         continue
                    
#                     header = [str(h or '').strip() for h in tbl[0]]
#                     rows = [[str(c or '').strip() for c in r] for r in tbl[1:]]
                    
#                     try:
#                         df = pd.DataFrame(rows, columns=header)
#                     except Exception:
#                         continue

#                     # Map headers to semantic targets
#                     target_map = {
#                         'segment': ['segment', 'business', 'business segment', 'unit', 'category'],
#                         'gov': ['gov', 'gross order value'],
#                         'revenue': ['revenue', 'sales'],
#                         'yoy': ['yoy', 'yoy growth', 'growth %', 'yo y'],
#                         'orders': ['orders'],
#                         'aov': ['aov', 'avg order value', 'average order value'],
#                     }
                    
#                     col_map = {}
#                     for tgt, syns in target_map.items():
#                         best_col = None
#                         best_score = 0.0
#                         for col in df.columns:
#                             score = _column_match_score(str(col), syns)
#                             # Also allow patterns like 'GOV (Cr ₹)'
#                             if not score and tgt in ('gov', 'yoy') and re.search(r"\bGOV\b", str(col), re.I):
#                                 score = 0.9
#                             if score > best_score:
#                                 best_score = score
#                                 best_col = col
#                         col_map[tgt] = best_col if best_score >= 0.6 else None

#                     seg_col = col_map.get('segment')
#                     if not seg_col:
#                         continue
                    
#                     for _, r in df.iterrows():
#                         seg_text = str(r.get(seg_col, '')).strip()
#                         if not seg_text:
#                             continue
                        
#                         section = _find_section_key(seg_text, ZOMATO_SECTIONS)
#                         if not section:
#                             continue
                        
#                         # Extract numeric values
#                         if col_map.get('gov'):
#                             gv = _to_number(r.get(col_map['gov']))
#                             if gv is not None:
#                                 segments[section]['GOV'] = gv
#                         if col_map.get('revenue'):
#                             rv = _to_number(r.get(col_map['revenue']))
#                             if rv is not None:
#                                 segments[section]['Revenue'] = rv
#                         if col_map.get('yoy'):
#                             yy = str(r.get(col_map['yoy']) or '').strip()
#                             if yy:
#                                 segments[section]['YoYGrowth'] = yy
#                         if col_map.get('orders'):
#                             od = _to_number(r.get(col_map['orders']))
#                             if od is not None:
#                                 segments[section]['Orders'] = od
#                         if col_map.get('aov'):
#                             av = _to_number(r.get(col_map['aov']))
#                             if av is not None:
#                                 segments[section]['AOV'] = av
#     except Exception:
#         pass

#     return {
#         'company': 'Zomato',
#         'reportDate': None,
#         'keyMetrics': segments,
#         'esg': esg,
#     }


# def analyze_zomato(structured: dict) -> dict:
#     """Analyze Zomato data and generate insights."""
#     km = structured.get('keyMetrics', {})
#     trends = {
#         'positive': [],
#         'negative': [],
#     }
#     recs = []

#     # Detect fastest growth
#     growths = []
#     for seg, vals in km.items():
#         yoy = vals.get('YoYGrowth')
#         if yoy and isinstance(yoy, str) and yoy.strip('%').replace('+', '').replace('-', '').isdigit():
#             try:
#                 growths.append((seg, float(yoy.strip('%').replace('+', ''))))
#             except Exception:
#                 pass
    
#     if growths:
#         growths.sort(key=lambda x: x[1], reverse=True)
#         top = growths[0]
#         seg_label = {
#             'foodDelivery': 'Food Delivery',
#             'quickCommerce': 'Quick Commerce',
#             'goingOut': 'Going-Out',
#             'b2bSupplies': 'B2B Supplies (Hyperpure)'
#         }.get(top[0], top[0])
#         trends['positive'].append(f"{seg_label} shows highest YoY growth ({top[1]:.0f}%).")

#     # Example risk: Hyperpure margin concern if Revenue present but YoY negative
#     b2b = km.get('b2bSupplies', {})
#     if b2b.get('Revenue') is not None:
#         yoy = b2b.get('YoYGrowth')
#         if yoy and '-' in str(yoy):
#             trends['negative'].append('Hyperpure (B2B) growth appears negative YoY.')

#     # Recommendations examples
#     if any(s for s, g in growths if s == 'quickCommerce'):
#         recs.append('Expand Quick Commerce store footprint aggressively.')
#     if b2b.get('Revenue'):
#         recs.append('Optimize Hyperpure supply chain to improve profitability.')
#     if structured.get('esg', {}).get('NetZeroTarget'):
#         recs.append('Leverage ESG leadership (Net Zero commitment) as a competitive advantage.')

#     return {
#         'company': structured.get('company', 'Zomato'),
#         'reportDate': structured.get('reportDate') or '2023-24',
#         'keyMetrics': km,
#         'trends': trends,
#         'recommendations': recs,
#     }


# # ---------- Django Views ----------

# def upload_view(request):
#     """Handle file upload and financial analysis."""
#     if request.method == 'POST':
#         form = UploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             uploaded = form.cleaned_data['file']
#             # Save to media folder
#             filename = uploaded.name
#             save_path = os.path.join(settings.MEDIA_ROOT, filename)
#             os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
#             with open(save_path, 'wb+') as dest:
#                 for chunk in uploaded.chunks():
#                     dest.write(chunk)

#             # Parse file
#             ext = os.path.splitext(filename)[1].lower()
#             try:
#                 if ext == '.pdf':
#                     extracted = extract_metrics_from_pdf(save_path)
#                 elif ext == '.csv':
#                     extracted = extract_metrics_from_csv_chunked(save_path)
#                 elif ext in ['.xlsx', '.xls']:
#                     engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
#                     df = pd.read_excel(save_path, engine=engine)
#                     extracted = map_columns_to_metrics(df)
#                 else:
#                     raise ValueError('Unsupported file extension.')
#             except Exception as e:
#                 context = {
#                     'form': form,
#                     'error': f'Failed to parse file: {e}'
#                 }
#                 return render(request, 'core/upload.html', context)

#             # Analysis
#             result = analyze_financials(extracted)
#             if isinstance(result, JsonResponse):
#                 context = {'form': form, 'error': result.content.decode()}
#                 return render(request, 'core/upload.html', context)
            
#             result['source_file'] = filename
#             request.session['last_result'] = result

#             return render(request, 'core/result.html', {'result': result})
#         else:
#             return render(request, 'core/upload.html', {'form': form})

#     # GET request
#     form = UploadForm()
#     return render(request, 'core/upload.html', {'form': form})


# def zomato_upload_view(request):
#     """Handle Zomato Annual Report uploads and analysis."""
#     error = None
#     result = None

#     def render_result(result):
#         request.session.pop("zomato_result", None)
#         return render(request, "core/zomato_result.html", {"result": result})

#     if request.method == "POST":
#         form = UploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             try:
#                 file = request.FILES["file"]
#                 file_bytes = file.read()
                
#                 if not file.name.lower().endswith(".pdf"):
#                     error = "Only PDF analysis supported for Zomato right now."
#                     return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
#                 # Extract text from PDF
#                 extracted_text = extract_text_from_pdf(file_bytes)
#                 if extracted_text.startswith("⚠️"):
#                     error = extracted_text
#                     return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
#                 # Call AI model
#                 ai_prompt = f"""
#                 You are a financial analyst. Analyze the following Zomato Annual Report text
#                 and return JSON with:
#                 - keyMetrics (foodDelivery, quickCommerce, goingOut, b2bSupplies)
#                 - trends (positive, negative)
#                 - recommendations
#                 Keep JSON structure clean and machine-readable.

#                 Report Text:
#                 {extracted_text}
#                 """
                
#                 ai_response = call_ai_model(ai_prompt)
#                 if isinstance(ai_response, dict) and ai_response.get("error"):
#                     error = ai_response["error"]
#                     return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
#                 # Parse AI response
#                 try:
#                     result = ai_response if isinstance(ai_response, dict) else json.loads(ai_response)
#                 except Exception:
#                     error = "AI response was not valid JSON."
#                     return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
#                 # Save to session and render result
#                 request.session["zomato_result"] = result
#                 request.session.modified = True
#                 return render_result(result)
                
#             except Exception as e:
#                 error = f"Error while processing file: {str(e)}"
#                 return render(request, "core/zomato_upload.html", {"form": form, "error": error})
#         else:
#             error = "Invalid form submission."
#             return render(request, "core/zomato_upload.html", {"form": form, "error": error})
#     else:
#         form = UploadForm()
#         # If result exists in session, show it and clear after display
#         result = request.session.pop("zomato_result", None)
#         if result:
#             return render_result(result)
    
#     return render(request, "core/zomato_upload.html", {"form": form, "error": error})


# def export_json_view(request):
#     """Export financial analysis results as JSON."""
#     data = request.session.get('last_result')
#     if not data:
#         return redirect(reverse('core:upload'))

#     json_bytes = json.dumps(data, indent=2).encode('utf-8')
#     response = HttpResponse(json_bytes, content_type='application/json')
#     ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
#     filename = f"ai_cfo_report_{ts}.json"
#     response['Content-Disposition'] = f'attachment; filename="{filename}"'
#     return response


# def zomato_export_json_view(request):
#     """Export Zomato analysis results as JSON."""
#     data = request.session.get('zomato_result')
#     if not data:
#         return redirect(reverse('core:zomato_upload'))
    
#     content = json.dumps(data, indent=2).encode('utf-8')
#     response = HttpResponse(content, content_type='application/json')
#     ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
#     response['Content-Disposition'] = f'attachment; filename="zomato_report_{ts}.json"'
#     return response


# def zomato_export_pdf_view(request):
#     """Export Zomato analysis results as PDF."""
#     data = request.session.get('zomato_result')
#     if not data:
#         return redirect(reverse('core:zomato_upload'))

#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=A4, title="Zomato Annual Report Insights")
#     styles = getSampleStyleSheet()
#     story = []

#     story.append(Paragraph("Zomato Annual Report Insights", styles['Title']))
#     story.append(Paragraph(f"Generated: {data.get('generated_at','')} • Source: {data.get('source_file','')}", styles['Normal']))
#     story.append(Spacer(1, 18))

#     # Executive Summary
#     story.append(Paragraph("Executive Summary", styles['Heading2']))
#     story.append(Paragraph("This report summarizes key metrics, trends, risks, and opportunities derived from the annual report.", styles['Normal']))
#     story.append(Spacer(1, 12))

#     # Key Metrics by segment
#     story.append(Paragraph("Key Metrics by Segment", styles['Heading2']))
#     km = data.get('keyMetrics', {})
#     seg_names = {
#         'foodDelivery': 'Food Delivery',
#         'quickCommerce': 'Quick Commerce',
#         'goingOut': 'Going-Out',
#         'b2bSupplies': 'B2B Supplies (Hyperpure)'
#     }
    
#     for seg_key, seg_label in seg_names.items():
#         vals = km.get(seg_key, {})
#         if not vals:
#             continue
        
#         story.append(Paragraph(seg_label, styles['Heading3']))
#         td = [["Metric", "Value"]]
#         for k in ['GOV', 'Revenue', 'YoYGrowth', 'Orders', 'AOV']:
#             if vals.get(k) is not None:
#                 v = vals.get(k)
#                 if k == 'YoYGrowth':
#                     disp = str(v)
#                 else:
#                     try:
#                         disp = f"{float(v):,.2f}"
#                     except Exception:
#                         disp = str(v)
#                 td.append([k, disp])
        
#         t = Table(td, hAlign='LEFT')
#         t.setStyle(TableStyle([
#             ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
#             ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
#             ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
#             ('ALIGN', (1,1), (1,-1), 'RIGHT'),
#         ]))
#         story += [t, Spacer(1, 12)]

#     # Trends
#     story.append(Paragraph("Key Trends", styles['Heading2']))
#     trends = data.get('trends', {})
#     pos_style = ParagraphStyle('Positive', parent=styles['Normal'], textColor=colors.green)
#     neg_style = ParagraphStyle('Negative', parent=styles['Normal'], textColor=colors.red)
    
#     for item in trends.get('positive', []):
#         story.append(Paragraph(f"• {item}", pos_style))
#     for item in trends.get('negative', []):
#         story.append(Paragraph(f"• {item}", neg_style))
#     story.append(Spacer(1, 12))

#     # Recommendations
#     story.append(Paragraph("Recommendations", styles['Heading2']))
#     for r in data.get('recommendations', []):
#         story.append(Paragraph(f"• {r}", styles['Normal']))

#     doc.build(story)
#     pdf = buffer.getvalue()
#     buffer.close()
    
#     response = HttpResponse(pdf, content_type='application/pdf')
#     ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
#     response['Content-Disposition'] = f'attachment; filename="zomato_report_{ts}.pdf"'
#     return response


# def cfo_export_pdf_view(request):
#     """Export CFO analysis results as PDF."""
#     data = request.session.get('last_result')
#     if not data:
#         return redirect(reverse('core:upload'))

#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=A4, title="AI CFO Assistant Report")
#     styles = getSampleStyleSheet()
#     story = []

#     # Title & Metadata
#     story.append(Paragraph("AI CFO Assistant Report", styles['Title']))
#     story.append(Paragraph(f"Generated: {data.get('generated_at','')} • Source: {data.get('source_file','')}", styles['Normal']))
#     story.append(Spacer(1, 18))

#     # Executive Summary
#     story.append(Paragraph("Executive Summary", styles['Heading2']))
#     exec_summary = data.get("executive_summary", "No executive summary available.")
#     story.append(Paragraph(exec_summary, styles['Normal']))
#     story.append(Spacer(1, 12))

#     # Key Metrics Table
#     story.append(Paragraph("Key Metrics", styles['Heading2']))
#     metrics = data.get("metrics", {})
#     table_data = [["Metric", "Value"]]
    
#     for k, v in metrics.items():
#         if v is None:
#             disp = "—"
#         else:
#             if isinstance(v, (int, float)):
#                 if "Margin" in k or "ROE" in k or "ROA" in k:
#                     disp = f"{v:.2f}%"
#                 elif "Runway" in k:
#                     disp = f"{v:.1f} months"
#                 else:
#                     disp = f"{v:,.2f}"
#             else:
#                 disp = str(v)
#         table_data.append([k, disp])

#     tbl = Table(table_data, hAlign='LEFT')
#     tbl.setStyle(TableStyle([
#         ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
#         ('TEXTCOLOR', (0,0), (-1,0), colors.black),
#         ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
#         ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
#         ('ALIGN', (1,1), (1,-1), 'RIGHT'),
#     ]))
#     story.append(tbl)
#     story.append(Spacer(1, 18))

#     # Risks
#     story.append(Paragraph("Detected Risks", styles['Heading2']))
#     risks = data.get("risks", [])
#     if risks:
#         for r in risks:
#             level = r.get("level", "Unknown")
#             style = styles['Normal']
#             if level == "High":
#                 style = ParagraphStyle("HighRisk", parent=styles['Normal'], textColor=colors.red)
#             elif level == "Medium":
#                 style = ParagraphStyle("MedRisk", parent=styles['Normal'], textColor=colors.orange)
#             elif level == "Low":
#                 style = ParagraphStyle("LowRisk", parent=styles['Normal'], textColor=colors.green)

#             story.append(Paragraph(f"⚠️ {level} — {r.get('title','')}", style))
#             story.append(Paragraph(r.get("description",""), styles['Normal']))
#             story.append(Spacer(1, 10))
#     else:
#         story.append(Paragraph("No major risks detected.", styles['Normal']))

#     story.append(Spacer(1, 18))

#     # Recommendations
#     story.append(Paragraph("Recommendations", styles['Heading2']))
#     recs = data.get("recommendations", [])
#     if recs:
#         for rec in recs:
#             story.append(Paragraph(f"{rec.get('priority', '')}. {rec.get('action','')}", styles['Heading3']))
#             story.append(Paragraph(rec.get('details',''), styles['Normal']))
#             story.append(Spacer(1, 10))
#     else:
#         story.append(Paragraph("No recommendations generated.", styles['Normal']))

#     story.append(Spacer(1, 18))

#     # Missing Data
#     story.append(Paragraph("Missing Data", styles['Heading2']))
#     missing = data.get("missing_data", [])
#     if missing:
#         for item in missing:
#             story.append(Paragraph(f"• {item}", styles['Normal']))
#     else:
#         story.append(Paragraph("All key data points were provided.", styles['Normal']))

#     # Build PDF
#     doc.build(story)
#     pdf = buffer.getvalue()
#     buffer.close()

#     response = HttpResponse(pdf, content_type="application/pdf")
#     ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
#     filename = f"ai_cfo_report_{ts}.pdf"
#     response["Content-Disposition"] = f'attachment; filename="{filename}"'
#     return response

import json
import os
import re
from io import BytesIO
from datetime import datetime
import google.generativeai as genai
import pandas as pd
import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from .forms import UploadForm
from PyPDF2 import PdfReader


# ---------- Configuration ----------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

METRIC_SYNONYMS = {
    'revenue': ['revenue', 'sales', 'turnover', 'total revenue'],
    'expenses': ['expense', 'expenses', 'operating expenses', 'opex', 'total expenses', 'costs'],
    'cash_reserves': ['cash', 'cash reserves', 'cash balance', 'cash and cash equivalents'],
    'assets': ['assets', 'total assets'],
    'liabilities': ['liabilities', 'total liabilities'],
    'equity': ['equity', 'shareholder equity', "shareholders' equity", 'net assets'],
    'debt': ['debt', 'loans', 'borrowings', 'total debt'],
}

ZOMATO_SECTIONS = {
    'foodDelivery': ['food delivery', 'delivery'],
    'quickCommerce': ['quick commerce', 'blinkit'],
    'goingOut': ['going-out', 'going out', 'dining out', 'dining'],
    'b2bSupplies': ['b2b supplies', 'hyperpure']
}


# ---------- Helper Functions ----------

def call_ai_model(prompt, model="gemini-1.5-flash"):
    """
    Calls Gemini Pro with a strict JSON prompt and validates the output.
    Always returns a Python dict or error dict.
    """
    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        text = response.text.strip() if hasattr(response, "text") else str(response)
        # Clean markdown JSON blocks if present
        if text.startswith("```"):
            text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
        # Try parsing JSON
        try:
            return json.loads(text)
        except Exception:
            return {"error": "AI response was not valid JSON", "raw": text}
    except Exception as e:
        return {"error": f"AI call failed: {e}"}


def normalize_col(name: str) -> str:
    """Normalize column name for comparison."""
    return str(name).strip().lower()


def _column_match_score(col_name: str, synonyms: list[str]) -> float:
    """Compute a simple fuzzy score [0..1] based on case-insensitive substring and similarity."""
    from difflib import SequenceMatcher
    n = normalize_col(col_name)
    best = 0.0
    
    for syn in synonyms:
        syn_norm = syn.lower()
        # Exact match
        if syn_norm == n:
            return 1.0
        # Substring match
        if syn_norm in n or n in syn_norm:
            best = max(best, 0.8)
        # Similarity ratio
        ratio = SequenceMatcher(None, n, syn_norm).ratio()
        best = max(best, ratio)
    
    return best


def _to_number(s) -> float | None:
    """Convert string to number, handling currency symbols and formatting."""
    if s is None or s == '':
        return None
    
    s = str(s).strip()
    if not s:
        return None
    
    # Check for negative indicators
    neg = s.startswith('-') or s.startswith('(')
    
    # Remove common non-numeric characters
    s = re.sub(r'[^\d,.\-+()]', '', s)
    s = s.replace('(', '').replace(')', '')
    
    # Handle comma/dot formatting
    if ',' in s and '.' in s:
        s = s.replace(',', '')
    elif s.count(',') >= 1 and s.count('.') == 0:
        s = s.replace(',', '')
    
    try:
        num = float(s)
        return -num if neg else num
    except Exception:
        return None


def _relevant_pdf_pages(pdf, keywords: list[str], max_pages: int = 50) -> list[int]:
    """Find pages containing relevant keywords."""
    relevant_pages = []
    
    for i, page in enumerate(pdf.pages[:max_pages]):
        try:
            text = page.extract_text() or ""
            text_lower = text.lower()
            
            # Check if any keyword appears in the page
            if any(kw.lower() in text_lower for kw in keywords):
                relevant_pages.append(i)
        except Exception:
            continue
    
    return relevant_pages


def safe_float(val):
    """Safely convert value to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.
    Accepts either a file path (str) or bytes.
    Returns extracted text or error string.
    """
    text = ""
    try:
        if isinstance(file, str):
            # File path
            reader = PdfReader(file)
        elif isinstance(file, bytes):
            # Bytes - wrap in BytesIO to make it file-like
            from io import BytesIO
            reader = PdfReader(BytesIO(file))
        else:
            return "⚠️ Invalid file input."
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        return f"⚠️ Error reading PDF: {e}"
    return text


def map_columns_to_metrics(df):
    """Map DataFrame columns to financial metrics."""
    mapped = {k: None for k in METRIC_SYNONYMS.keys()}
    
    for metric, synonyms in METRIC_SYNONYMS.items():
        best_col = None
        best_score = 0.0
        
        for col in df.columns:
            score = _column_match_score(str(col), synonyms)
            if score > best_score and score >= 0.6:
                best_score = score
                best_col = col
        
        if best_col is not None:
            # Get the first non-null numeric value from this column
            series = pd.to_numeric(df[best_col], errors='coerce')
            first_valid = series.dropna().iloc[0] if not series.dropna().empty else None
            mapped[metric] = first_valid
    
    return mapped


def extract_metrics_from_csv_chunked(file_path: str, chunk_size: int = 1000) -> dict:
    """Extract metrics from CSV using chunked processing for large files."""
    mapped = {k: None for k in METRIC_SYNONYMS.keys()}
    
    try:
        # Read first chunk to get column structure
        first_chunk = pd.read_csv(file_path, nrows=chunk_size)
        return map_columns_to_metrics(first_chunk)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return mapped


def extract_metrics_from_pdf(file_path: str) -> dict:
    """Extract metrics from a PDF. Try tables on relevant pages first; fallback to text scanning."""
    metrics = {k: None for k in METRIC_SYNONYMS.keys()}
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # Build keyword list from all metric synonyms
            keywords = []
            for v in METRIC_SYNONYMS.values():
                keywords.extend(v)
            
            pages_idx = _relevant_pdf_pages(pdf, keywords)[:20]
            
            # Try to extract from tables first
            for i in pages_idx:
                page = pdf.pages[i]
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                
                for tbl in tables:
                    if not tbl or len(tbl) < 2 or len(tbl[0]) < 2:
                        continue
                    
                    header = [h if h is not None else '' for h in tbl[0]]
                    rows = [[c if c is not None else '' for c in r] for r in tbl[1:]]
                    
                    # Limit excessive rows for performance
                    if len(rows) > 1000:
                        rows = rows[:1000]
                    
                    try:
                        df = pd.DataFrame(rows, columns=header)
                        extracted = map_columns_to_metrics(df)
                        
                        # Update metrics with non-null values
                        for key, value in extracted.items():
                            if value is not None and metrics[key] is None:
                                metrics[key] = value
                    except Exception:
                        continue
            
            # Fallback to text extraction if no metrics found
            if all(v is None for v in metrics.values()):
                full_text = ""
                for i in pages_idx:
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    full_text += text
                
                # Text pattern matching
                for key, synonyms in METRIC_SYNONYMS.items():
                    if metrics[key] is not None:
                        continue
                    
                    for syn in synonyms:
                        syn_esc = re.escape(syn)
                        pattern = rf"(?i)\b{syn_esc}\b\s*[:\-]?\s*\(?[-+]?\$?[\d,.]+\)?"
                        for m in re.finditer(pattern, full_text):
                            num_match = re.search(r"\(?[-+]?\$?[\d,.]+\)?$", m.group(0))
                            if num_match:
                                val = _to_number(num_match.group(0))
                                if val is not None:
                                    metrics[key] = val
                                    break
                        if metrics[key] is not None:
                            break
    
    except Exception as e:
        print(f"Error extracting from PDF: {e}")
    
    return metrics


def _find_section_key(text: str, section_map: dict) -> str | None:
    """Find which section a text belongs to."""
    t = text.lower()
    for key, synonyms in section_map.items():
        for s in synonyms:
            if s in t:
                return key
    return None


# ---------- Main Analysis Function ----------

@csrf_exempt
def analyze_financials(request):
    """
    Analyze financials from POSTed JSON or dict.
    Returns JsonResponse or dict.
    """
    if isinstance(request, dict):
        data = request
    else:
        if request.method != "POST":
            return JsonResponse({"error": "Only POST allowed"}, status=405)
        try:
            data = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Extract and validate input
    company = data.get("company", "Unknown Company")
    period = data.get("period", "Not specified")
    revenue = safe_float(data.get("revenue", 0))
    expenses = safe_float(data.get("expenses", 0))
    net_income = revenue - expenses if revenue and expenses else None
    burn_rate = safe_float(data.get("monthly_burn_rate", 0))
    debt = safe_float(data.get("debt", 0))
    equity = safe_float(data.get("equity", 0))

    metrics = {
        "Revenue": revenue,
        "Expenses": expenses,
        "Net Income": net_income if net_income is not None else "Data not provided",
        "Monthly Burn Rate": burn_rate if burn_rate else "Data not provided",
        "Runway (Months)": (
            round(revenue / burn_rate, 1) if burn_rate and burn_rate > 0 and revenue > 0 else "Not applicable (profitable or no burn)"
        ),
        "Profit Margin (%)": (
            round((net_income / revenue) * 100, 2) if revenue > 0 and net_income is not None else "Data not provided"
        ),
        "Debt-to-Equity Ratio": (
            round(debt / equity, 2) if equity > 0 else "Data not provided"
        ),
    }

    # Risks detection
    risks = []
    if isinstance(metrics["Profit Margin (%)"], (int, float)) and metrics["Profit Margin (%)"] < 5:
        risks.append({
            "level": "High",
            "title": "Low Profitability",
            "description": "Profit margins are below 5%, indicating potential cost or pricing challenges."
        })
    if isinstance(metrics["Runway (Months)"], (int, float)) and metrics["Runway (Months)"] < 6:
        risks.append({
            "level": "High",
            "title": "Short Cash Runway",
            "description": "Cash reserves may not sustain operations beyond 6 months at current burn rate."
        })
    if isinstance(metrics["Debt-to-Equity Ratio"], (int, float)) and metrics["Debt-to-Equity Ratio"] > 2:
        risks.append({
            "level": "High",
            "title": "High Financial Leverage",
            "description": "Debt-to-Equity ratio above 2 indicates excessive reliance on debt."
        })
    if revenue > 0 and expenses > revenue:
        risks.append({
            "level": "Medium",
            "title": "Expenses Exceed Revenue",
            "description": "Operating expenses are higher than revenue, leading to negative net income."
        })
    if not risks:
        risks.append({
            "level": "Low",
            "title": "No critical risks detected",
            "description": "Financials appear stable, but continuous monitoring is advised."
        })

    recommendations = [
        {
            "priority": 1,
            "action": "Improve liquidity management",
            "details": "Maintain at least 9–12 months runway for financial safety."
        },
        {
            "priority": 2,
            "action": "Optimize cost structure",
            "details": "Review operating expenses and reduce non-essential spend to protect margins."
        },
        {
            "priority": 3,
            "action": "Balance capital structure",
            "details": "If debt levels are high, explore equity infusion or refinancing."
        }
    ]

    missing_data = []
    if not debt:
        missing_data.append("Debt details")
    if not equity:
        missing_data.append("Equity details")
    if not burn_rate:
        missing_data.append("Cash flow / burn rate")
    if not expenses:
        missing_data.append("Operating expenses breakdown")

    exec_summary = f"""
    For {company} during {period}, reported revenue was {revenue:,.0f} with expenses of {expenses:,.0f}.
    Net income stands at {net_income:,.0f} ({metrics['Profit Margin (%)']}% margin).
    Current runway is {metrics['Runway (Months)']} months and Debt-to-Equity is {metrics['Debt-to-Equity Ratio']}.
    Key focus areas: cost discipline, liquidity management, and capital structure optimization.
    """

    response = {
        "company": company,
        "period": period,
        "executive_summary": exec_summary.strip(),
        "metrics": metrics,
        "risks": risks,
        "recommendations": recommendations,
        "missing_data": missing_data
    }

    if not isinstance(request, dict):
        return JsonResponse(response, safe=False, status=200)
    return response


def parse_zomato_pdf(file_path: str) -> dict:
    """Extract structured data for Zomato annual report."""
    segments = {
        'foodDelivery': {},
        'quickCommerce': {},
        'goingOut': {},
        'b2bSupplies': {},
    }
    esg = {}

    try:
        with pdfplumber.open(file_path) as pdf:
            kw = ['food delivery', 'quick commerce', 'going-out', 'going out', 'hyperpure', 'b2b', 'gov', 'revenue', 'yoy']
            pages_idx = _relevant_pdf_pages(pdf, kw)[:20]
            
            for i in pages_idx:
                page = pdf.pages[i]
                tables = []
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    pass
                
                for tbl in tables:
                    if not tbl or len(tbl) < 2 or len(tbl[0]) < 2:
                        continue
                    
                    header = [str(h or '').strip() for h in tbl[0]]
                    rows = [[str(c or '').strip() for c in r] for r in tbl[1:]]
                    
                    try:
                        df = pd.DataFrame(rows, columns=header)
                    except Exception:
                        continue

                    # Map headers to semantic targets
                    target_map = {
                        'segment': ['segment', 'business', 'business segment', 'unit', 'category'],
                        'gov': ['gov', 'gross order value'],
                        'revenue': ['revenue', 'sales'],
                        'yoy': ['yoy', 'yoy growth', 'growth %', 'yo y'],
                        'orders': ['orders'],
                        'aov': ['aov', 'avg order value', 'average order value'],
                    }
                    
                    col_map = {}
                    for tgt, syns in target_map.items():
                        best_col = None
                        best_score = 0.0
                        for col in df.columns:
                            score = _column_match_score(str(col), syns)
                            # Also allow patterns like 'GOV (Cr ₹)'
                            if not score and tgt in ('gov', 'yoy') and re.search(r"\bGOV\b", str(col), re.I):
                                score = 0.9
                            if score > best_score:
                                best_score = score
                                best_col = col
                        col_map[tgt] = best_col if best_score >= 0.6 else None

                    seg_col = col_map.get('segment')
                    if not seg_col:
                        continue
                    
                    for _, r in df.iterrows():
                        seg_text = str(r.get(seg_col, '')).strip()
                        if not seg_text:
                            continue
                        
                        section = _find_section_key(seg_text, ZOMATO_SECTIONS)
                        if not section:
                            continue
                        
                        # Extract numeric values
                        if col_map.get('gov'):
                            gv = _to_number(r.get(col_map['gov']))
                            if gv is not None:
                                segments[section]['GOV'] = gv
                        if col_map.get('revenue'):
                            rv = _to_number(r.get(col_map['revenue']))
                            if rv is not None:
                                segments[section]['Revenue'] = rv
                        if col_map.get('yoy'):
                            yy = str(r.get(col_map['yoy']) or '').strip()
                            if yy:
                                segments[section]['YoYGrowth'] = yy
                        if col_map.get('orders'):
                            od = _to_number(r.get(col_map['orders']))
                            if od is not None:
                                segments[section]['Orders'] = od
                        if col_map.get('aov'):
                            av = _to_number(r.get(col_map['aov']))
                            if av is not None:
                                segments[section]['AOV'] = av
    except Exception:
        pass

    return {
        'company': 'Zomato',
        'reportDate': None,
        'keyMetrics': segments,
        'esg': esg,
    }


def analyze_zomato(structured: dict) -> dict:
    """Analyze Zomato data and generate insights."""
    km = structured.get('keyMetrics', {})
    trends = {
        'positive': [],
        'negative': [],
    }
    recs = []

    # Detect fastest growth
    growths = []
    for seg, vals in km.items():
        yoy = vals.get('YoYGrowth')
        if yoy and isinstance(yoy, str) and yoy.strip('%').replace('+', '').replace('-', '').isdigit():
            try:
                growths.append((seg, float(yoy.strip('%').replace('+', ''))))
            except Exception:
                pass
    
    if growths:
        growths.sort(key=lambda x: x[1], reverse=True)
        top = growths[0]
        seg_label = {
            'foodDelivery': 'Food Delivery',
            'quickCommerce': 'Quick Commerce',
            'goingOut': 'Going-Out',
            'b2bSupplies': 'B2B Supplies (Hyperpure)'
        }.get(top[0], top[0])
        trends['positive'].append(f"{seg_label} shows highest YoY growth ({top[1]:.0f}%).")

    # Example risk: Hyperpure margin concern if Revenue present but YoY negative
    b2b = km.get('b2bSupplies', {})
    if b2b.get('Revenue') is not None:
        yoy = b2b.get('YoYGrowth')
        if yoy and '-' in str(yoy):
            trends['negative'].append('Hyperpure (B2B) growth appears negative YoY.')

    # Recommendations examples
    if any(s for s, g in growths if s == 'quickCommerce'):
        recs.append('Expand Quick Commerce store footprint aggressively.')
    if b2b.get('Revenue'):
        recs.append('Optimize Hyperpure supply chain to improve profitability.')
    if structured.get('esg', {}).get('NetZeroTarget'):
        recs.append('Leverage ESG leadership (Net Zero commitment) as a competitive advantage.')

    return {
        'company': structured.get('company', 'Zomato'),
        'reportDate': structured.get('reportDate') or '2023-24',
        'keyMetrics': km,
        'trends': trends,
        'recommendations': recs,
    }


# ---------- Django Views ----------

def upload_view(request):
    """Handle file upload and financial analysis."""
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = form.cleaned_data['file']
            # Save to media folder
            filename = uploaded.name
            save_path = os.path.join(settings.MEDIA_ROOT, filename)
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
            with open(save_path, 'wb+') as dest:
                for chunk in uploaded.chunks():
                    dest.write(chunk)

            # Parse file
            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext == '.pdf':
                    extracted = extract_metrics_from_pdf(save_path)
                elif ext == '.csv':
                    extracted = extract_metrics_from_csv_chunked(save_path)
                elif ext in ['.xlsx', '.xls']:
                    engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
                    df = pd.read_excel(save_path, engine=engine)
                    extracted = map_columns_to_metrics(df)
                else:
                    raise ValueError('Unsupported file extension.')
            except Exception as e:
                context = {
                    'form': form,
                    'error': f'Failed to parse file: {e}'
                }
                return render(request, 'core/upload.html', context)

            # Analysis
            result = analyze_financials(extracted)
            if isinstance(result, JsonResponse):
                context = {'form': form, 'error': result.content.decode()}
                return render(request, 'core/upload.html', context)
            
            result['source_file'] = filename
            request.session['last_result'] = result

            return render(request, 'core/result.html', {'result': result})
        else:
            return render(request, 'core/upload.html', {'form': form})

    # GET request
    form = UploadForm()
    return render(request, 'core/upload.html', {'form': form})


def zomato_upload_view(request):
    """Handle Zomato Annual Report uploads and analysis."""
    import logging
    logger = logging.getLogger(__name__)
    
    error = None
    result = None

    def render_result(result):
        request.session.pop("zomato_result", None)
        return render(request, "core/zomato_result.html", {"data": result})

    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                file = request.FILES["file"]
                file_bytes = file.read()
                logger.info(f"Processing file: {file.name}, size: {len(file_bytes)} bytes")
                
                if not file.name.lower().endswith(".pdf"):
                    error = "Only PDF analysis supported for Zomato right now."
                    return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
                # Extract text from PDF
                extracted_text = extract_text_from_pdf(file_bytes)
                logger.info(f"Extracted text length: {len(extracted_text)}")
                
                if extracted_text.startswith("⚠️"):
                    error = extracted_text
                    logger.error(f"PDF extraction failed: {error}")
                    return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
                # Call AI model
                ai_prompt = f"""
                You are a financial analyst. Analyze the following Zomato Annual Report text
                and return JSON with:
                - keyMetrics (foodDelivery, quickCommerce, goingOut, b2bSupplies)
                - trends (positive, negative)
                - recommendations
                Keep JSON structure clean and machine-readable.

                Report Text:
                {extracted_text[:5000]}
                """
                
                logger.info("Calling AI model...")
                ai_response = call_ai_model(ai_prompt)
                logger.info(f"AI Response type: {type(ai_response)}")
                logger.info(f"AI Response: {str(ai_response)[:500]}...")
                
                if isinstance(ai_response, dict) and ai_response.get("error"):
                    error = ai_response["error"]
                    logger.error(f"AI API Error: {error}")
                    return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
                # Parse AI response
                try:
                    result = ai_response if isinstance(ai_response, dict) else json.loads(ai_response)
                    logger.info("Successfully parsed AI response")
                except Exception as e:
                    error = f"AI response was not valid JSON: {str(e)}"
                    logger.error(f"JSON parsing failed: {error}")
                    logger.error(f"Raw AI response: {ai_response}")
                    return render(request, "core/zomato_upload.html", {"form": form, "error": error})
                
                # Save to session and render result
                request.session["zomato_result"] = result
                request.session.modified = True
                logger.info("Rendering result page")
                return render_result(result)
                
            except Exception as e:
                error = f"Error while processing file: {str(e)}"
                logger.error(f"Unexpected error: {error}", exc_info=True)
                return render(request, "core/zomato_upload.html", {"form": form, "error": error})
        else:
            error = "Invalid form submission."
            logger.warning(f"Form validation failed: {form.errors}")
            return render(request, "core/zomato_upload.html", {"form": form, "error": error})
    else:
        form = UploadForm()
        # If result exists in session, show it and clear after display
        result = request.session.pop("zomato_result", None)
        if result:
            logger.info("Displaying cached result from session")
            return render_result(result)
    
    return render(request, "core/zomato_upload.html", {"form": form, "error": error})


def export_json_view(request):
    """Export financial analysis results as JSON."""
    data = request.session.get('last_result')
    if not data:
        return redirect(reverse('core:upload'))

    json_bytes = json.dumps(data, indent=2).encode('utf-8')
    response = HttpResponse(json_bytes, content_type='application/json')
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    filename = f"ai_cfo_report_{ts}.json"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def zomato_export_json_view(request):
    """Export Zomato analysis results as JSON."""
    data = request.session.get('zomato_result')
    if not data:
        return redirect(reverse('core:zomato_upload'))
    
    content = json.dumps(data, indent=2).encode('utf-8')
    response = HttpResponse(content, content_type='application/json')
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    response['Content-Disposition'] = f'attachment; filename="zomato_report_{ts}.json"'
    return response


def zomato_export_pdf_view(request):
    """Export Zomato analysis results as PDF."""
    data = request.session.get('zomato_result')
    if not data:
        return redirect(reverse('core:zomato_upload'))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="Zomato Annual Report Insights")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Zomato Annual Report Insights", styles['Title']))
    story.append(Paragraph(f"Generated: {data.get('generated_at','')} • Source: {data.get('source_file','')}", styles['Normal']))
    story.append(Spacer(1, 18))

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Paragraph("This report summarizes key metrics, trends, risks, and opportunities derived from the annual report.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Key Metrics by segment
    story.append(Paragraph("Key Metrics by Segment", styles['Heading2']))
    km = data.get('keyMetrics', {})
    seg_names = {
        'foodDelivery': 'Food Delivery',
        'quickCommerce': 'Quick Commerce',
        'goingOut': 'Going-Out',
        'b2bSupplies': 'B2B Supplies (Hyperpure)'
    }
    
    for seg_key, seg_label in seg_names.items():
        vals = km.get(seg_key, {})
        if not vals:
            continue
        
        story.append(Paragraph(seg_label, styles['Heading3']))
        td = [["Metric", "Value"]]
        for k in ['GOV', 'Revenue', 'YoYGrowth', 'Orders', 'AOV']:
            if vals.get(k) is not None:
                v = vals.get(k)
                if k == 'YoYGrowth':
                    disp = str(v)
                else:
                    try:
                        disp = f"{float(v):,.2f}"
                    except Exception:
                        disp = str(v)
                td.append([k, disp])
        
        t = Table(td, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (1,1), (1,-1), 'RIGHT'),
        ]))
        story += [t, Spacer(1, 12)]

    # Trends
    story.append(Paragraph("Key Trends", styles['Heading2']))
    trends = data.get('trends', {})
    pos_style = ParagraphStyle('Positive', parent=styles['Normal'], textColor=colors.green)
    neg_style = ParagraphStyle('Negative', parent=styles['Normal'], textColor=colors.red)
    
    for item in trends.get('positive', []):
        story.append(Paragraph(f"• {item}", pos_style))
    for item in trends.get('negative', []):
        story.append(Paragraph(f"• {item}", neg_style))
    story.append(Spacer(1, 12))

    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    for r in data.get('recommendations', []):
        story.append(Paragraph(f"• {r}", styles['Normal']))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    
    response = HttpResponse(pdf, content_type='application/pdf')
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    response['Content-Disposition'] = f'attachment; filename="zomato_report_{ts}.pdf"'
    return response


def cfo_export_pdf_view(request):
    """Export CFO analysis results as PDF."""
    data = request.session.get('last_result')
    if not data:
        return redirect(reverse('core:upload'))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="AI CFO Assistant Report")
    styles = getSampleStyleSheet()
    story = []

    # Title & Metadata
    story.append(Paragraph("AI CFO Assistant Report", styles['Title']))
    story.append(Paragraph(f"Generated: {data.get('generated_at','')} • Source: {data.get('source_file','')}", styles['Normal']))
    story.append(Spacer(1, 18))

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    exec_summary = data.get("executive_summary", "No executive summary available.")
    story.append(Paragraph(exec_summary, styles['Normal']))
    story.append(Spacer(1, 12))

    # Key Metrics Table
    story.append(Paragraph("Key Metrics", styles['Heading2']))
    metrics = data.get("metrics", {})
    table_data = [["Metric", "Value"]]
    
    for k, v in metrics.items():
        if v is None:
            disp = "—"
        else:
            if isinstance(v, (int, float)):
                if "Margin" in k or "ROE" in k or "ROA" in k:
                    disp = f"{v:.2f}%"
                elif "Runway" in k:
                    disp = f"{v:.1f} months"
                else:
                    disp = f"{v:,.2f}"
            else:
                disp = str(v)
        table_data.append([k, disp])

    tbl = Table(table_data, hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,1), (1,-1), 'RIGHT'),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 18))

    # Risks
    story.append(Paragraph("Detected Risks", styles['Heading2']))
    risks = data.get("risks", [])
    if risks:
        for r in risks:
            level = r.get("level", "Unknown")
            style = styles['Normal']
            if level == "High":
                style = ParagraphStyle("HighRisk", parent=styles['Normal'], textColor=colors.red)
            elif level == "Medium":
                style = ParagraphStyle("MedRisk", parent=styles['Normal'], textColor=colors.orange)
            elif level == "Low":
                style = ParagraphStyle("LowRisk", parent=styles['Normal'], textColor=colors.green)

            story.append(Paragraph(f"⚠️ {level} — {r.get('title','')}", style))
            story.append(Paragraph(r.get("description",""), styles['Normal']))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No major risks detected.", styles['Normal']))

    story.append(Spacer(1, 18))

    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    recs = data.get("recommendations", [])
    if recs:
        for rec in recs:
            story.append(Paragraph(f"{rec.get('priority', '')}. {rec.get('action','')}", styles['Heading3']))
            story.append(Paragraph(rec.get('details',''), styles['Normal']))
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No recommendations generated.", styles['Normal']))

    story.append(Spacer(1, 18))

    # Missing Data
    story.append(Paragraph("Missing Data", styles['Heading2']))
    missing = data.get("missing_data", [])
    if missing:
        for item in missing:
            story.append(Paragraph(f"• {item}", styles['Normal']))
    else:
        story.append(Paragraph("All key data points were provided.", styles['Normal']))

    # Build PDF
    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()

    response = HttpResponse(pdf, content_type="application/pdf")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"ai_cfo_report_{ts}.pdf"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response