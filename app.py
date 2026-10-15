import os
import json
from pathlib import Path
from typing import List
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from pypdf import PdfReader
import docx

from google import genai
from google.genai import types

# =====================================================================
# 1. CONFIGURATION & LOCAL PERSISTENCE FOR RESUMES
# =====================================================================
st.set_page_config(
    page_title="Career Research & ATS Matcher Agent",
    page_icon="🎯",
    layout="wide",
)

RESUME_STORE_PATH = Path("saved_resumes.json")


def load_saved_resumes() -> dict:
    if RESUME_STORE_PATH.exists():
        try:
            return json.loads(RESUME_STORE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cs_resume": "", "pm_resume": ""}


def save_resumes_to_disk(cs_text: str, pm_text: str):
    RESUME_STORE_PATH.write_text(
        json.dumps({"cs_resume": cs_text, "pm_resume": pm_text}, indent=2),
        encoding="utf-8",
    )


def extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in ["docx", "doc"]:
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def fetch_clean_webpage_text(url: str, max_chars: int = 8000) -> str:
    """Attempts to scrape the full job description from a direct ATS URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CareerAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return text[:max_chars]
    except Exception:
        pass
    return ""


# =====================================================================
# 2. PYDANTIC SCHEMAS FOR STRUCTURED ATS OUTPUT
# =====================================================================
class RubricBreakdown(BaseModel):
    hard_skills_score: int = Field(description="Score out of 100 for Hard Skills & Software Stack (40% weight)")
    responsibilities_score: int = Field(description="Score out of 100 for Core Responsibilities & Lifecycle Keywords (35% weight)")
    experience_scope_score: int = Field(description="Score out of 100 for Experience & Scope Match (15% weight)")
    metrics_impact_score: int = Field(description="Score out of 100 for Quantifiable Impact & Metrics (10% weight)")


class BulletRewrite(BaseModel):
    target_section_or_role: str = Field(description="Which role or section on the candidate's resume to update")
    rewritten_bullet: str = Field(description="Ready-to-paste resume bullet incorporating missing ATS keywords and metrics")
    keywords_added: List[str] = Field(description="Exact ATS keywords injected into this bullet")


class JobAuditItem(BaseModel):
    company_name: str
    job_title: str
    location_work_style: str = Field(description="e.g., Remote (US) or Hybrid - New York, NY")
    compensation_range: str = Field(description="Base salary / OTE range listed or estimated")
    company_stage_note: str = Field(description="Verification that company is mature/established")
    direct_application_url: str
    selected_resume_track: str = Field(description="Either 'Customer Success Resume' or 'Project Management Resume'")
    initial_ats_score: int = Field(description="Initial weighted ATS score from 0 to 100")
    projected_ats_score: int = Field(description="Projected ATS score after applying suggested edits (target 90-95)")
    one_sentence_summary: str = Field(description="1-sentence executive summary of the ATS audit and path to 90-95%")
    rubric_breakdown: RubricBreakdown
    matched_keywords: List[str] = Field(description="Exact keywords from the job post already present in the resume")
    missing_critical_keywords: List[str] = Field(description="High-frequency keywords in the job post missing from the resume")
    tailored_bullet_rewrites: List[BulletRewrite] = Field(description="2 to 3 specific resume bullet edits to reach 90-95% match")


class BatchJobAuditResponse(BaseModel):
    jobs: List[JobAuditItem]


# =====================================================================
# 3. GEMINI API PIPELINE (SEARCH GROUNDING + DEEP ATS AUDIT)
# =====================================================================
def run_search_and_audit_pipeline(
    api_key: str,
    model_name: str,
    cs_resume: str,
    pm_resume: str,
    num_jobs: int,
    min_salary: int,
    days_posted: int,
    roles_list: List[str],
    locations_list: List[str],
    exclusions_text: str,
) -> BatchJobAuditResponse:
    client = genai.Client(api_key=api_key)

    # -----------------------------------------------------------------
    # STAGE 1: Live Web Search with Google Search Grounding
    # -----------------------------------------------------------------
    search_prompt = f"""
    You are an autonomous career research agent. Search the live web and direct ATS platforms
    (such as job-boards.greenhouse.io, jobs.lever.co, jobs.ashbyhq.com, myworkdayjobs.com, and direct
    company career pages) for {num_jobs} active job openings posted within the last {days_posted} days.

    STRICT CRITERIA:
    1. Target Job Titles: {", ".join(roles_list)}
    2. STRICT TITLE EXCLUSION: Exclude ALL "Senior", "Sr.", "Lead", "Principal", "Director", or "VP" titles.
       Only include Associate or Manager level roles.
    3. EXPERIENCE CAP: Exclude any role requiring 5 or more years of experience in Customer Success,
       Account Management, or Consultancy (target 1-4 years required experience).
    4. INDUSTRY & TOOL EXCLUSIONS: {exclusions_text}
    5. Location & Work Style: {", ".join(locations_list)}
    6. Compensation & Company Stage: Minimum base salary of ${min_salary:,}/year. Mature, established companies only (no early-stage startups).
    7. URL RULE: Strictly EXCLUDE aggregator links (LinkedIn, Indeed, Glassdoor, ZipRecruiter, BuiltIn).
       Provide ONLY direct company career page or direct ATS board URLs.

    For each of the {num_jobs} roles found, provide:
    - Company Name
    - Exact Job Title
    - Location / Work Style
    - Compensation Range
    - Direct Application URL
    - Detailed summary of required skills, tools, responsibilities, and qualifications from the posting.
    """

    search_response = client.models.generate_content(
        model=model_name,
        contents=search_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        ),
    )

    raw_search_findings = search_response.text or ""

    # Extract grounded URLs from metadata as backup reference
    grounded_urls = []
    if search_response.candidates and search_response.candidates[0].grounding_metadata:
        chunks = search_response.candidates[0].grounding_metadata.grounding_chunks or []
        for chunk in chunks:
            if chunk.web and chunk.web.uri:
                grounded_urls.append(f"- {chunk.web.title}: {chunk.web.uri}")
    grounded_urls_text = "\n".join(grounded_urls)

    # -----------------------------------------------------------------
    # STAGE 2: Structured ATS Resume Audit against CS & PM Resumes
    # -----------------------------------------------------------------
    audit_prompt = f"""
    You are an expert enterprise Applicant Tracking System (ATS) algorithm (modeled after Greenhouse, Lever, and Workday screening engines).

    Below are the candidate's two reference resumes:

    === CANDIDATE RESUME 1: CUSTOMER SUCCESS (CS) RESUME ===
    {cs_resume if cs_resume.strip() else "[No custom CS resume uploaded yet - evaluate against a standard mid-level CS Manager baseline with SaaS onboarding, retention, QBRs, and CRM experience]"}

    === CANDIDATE RESUME 2: PROJECT MANAGEMENT (PM) RESUME ===
    {pm_resume if pm_resume.strip() else "[No custom PM resume uploaded yet - evaluate against a standard mid-level PM baseline with Initiation, Planning, Execution, Monitoring, Closure, and cross-functional stakeholder alignment]"}

    === LIVE SEARCH RESULTS & JOB DESCRIPTIONS ===
    {raw_search_findings}

    === VERIFIED GROUNDED URLS FROM GOOGLE SEARCH ===
    {grounded_urls_text}

    TASK:
    Compile the discovered roles (up to {num_jobs} roles that strictly pass all exclusions: no Senior titles, <5 yrs required exp, no niche vertical tool mandates like HRIS/wealth management, >=${min_salary:,} base).

    For EACH role:
    1. Select whichever resume ('Customer Success Resume' or 'Project Management Resume') is the stronger natural fit.
    2. Score the selected resume like an ATS using this exact weighting:
       - Hard Skills & Software Stack (40%)
       - Core Responsibilities & Lifecycle Keywords (35%)
       - Experience & Scope Match (15%)
       - Quantifiable Impact & Metrics (10%)
    3. Provide the Initial ATS Score, Matched Keywords, Missing Critical Keywords, and 2-3 Ready-to-Paste Rewritten Resume Bullets that raise the score to 90%-95%.
    """

    structured_response = client.models.generate_content(
        model=model_name,
        contents=audit_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchJobAuditResponse,
            temperature=0.2,
        ),
    )

    return BatchJobAuditResponse.model_validate_json(structured_response.text)


def run_single_job_audit(
    api_key: str,
    model_name: str,
    cs_resume: str,
    pm_resume: str,
    job_url: str,
    job_text_override: str,
) -> JobAuditItem:
    client = genai.Client(api_key=api_key)
    scraped_text = fetch_clean_webpage_text(job_url) if job_url else ""
    combined_jd = f"URL: {job_url}\n\nScraped Page Content:\n{scraped_text}\n\nUser Pasted JD:\n{job_text_override}"

    prompt = f"""
    Act as an enterprise ATS screening engine. Audit the following single job posting against the candidate's two resumes.
    Automatically select whichever resume ('Customer Success Resume' or 'Project Management Resume') scores higher.

    === CUSTOMER SUCCESS RESUME ===
    {cs_resume if cs_resume.strip() else "[Standard mid-level Customer Success profile]"}

    === PROJECT MANAGEMENT RESUME ===
    {pm_resume if pm_resume.strip() else "[Standard mid-level Project Management profile]"}

    === TARGET JOB POSTING ===
    {combined_jd}

    Provide a complete ATS audit, weighted rubric breakdown, exact keyword gap matrix, and 3 ready-to-paste resume bullet rewrites to reach a 90%-95% match score.
    """

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JobAuditItem,
            temperature=0.2,
        ),
    )
    return JobAuditItem.model_validate_json(resp.text)


# =====================================================================
# 4. STREAMLIT USER INTERFACE
# =====================================================================
saved_data = load_saved_resumes()
if "cs_resume" not in st.session_state:
    st.session_state["cs_resume"] = saved_data.get("cs_resume", "")
if "pm_resume" not in st.session_state:
    st.session_state["pm_resume"] = saved_data.get("pm_resume", "")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get your key from aistudio.google.com",
    )
    model_choice = st.selectbox(
        "Gemini Model",
        options=["gemini-3.1-pro", "gemini-4", "gemini-3.1-pro-preview"],
        index=0,
        help="gemini-2.5-flash is fast and ideal for daily 15-job runs; gemini-2.5-pro offers deeper reasoning.",
    )

    st.divider()
    st.subheader("🎯 Search Criteria")
    num_jobs = st.slider("Number of Job Posts", min_value=5, max_value=20, value=15)
    days_posted = st.slider("Posted Within (Days)", min_value=1, max_value=14, value=7)
    min_salary = st.number_input("Minimum Base Salary ($)", value=90000, step=5000)

    default_roles = [
        "Onboarding Manager",
        "Onboarding Associate",
        "Customer Success Manager",
        "Customer Success Associate",
        "Implementation Manager",
        "Implementation Associate",
        "Project Manager",
        "Project Management Associate",
    ]
    selected_roles = st.multiselect("Target Titles", options=default_roles, default=default_roles)

    default_locations = ["Remote (US)", "Hybrid in New York City, NY", "Hybrid in Brooklyn, NY"]
    selected_locations = st.multiselect("Locations / Work Style", options=default_locations, default=default_locations)

    exclusions_text = st.text_area(
        "Strict Exclusions",
        value=(
            "Exclude Senior/Sr. titles. Exclude roles requiring 5+ years of CS or consulting experience. "
            "Exclude roles requiring specialized vertical industry/tool experience such as HRIS/payroll, "
            "wealth management, clinical EHR, or banking KYC/KYB."
        ),
        height=110,
    )

    st.divider()
    st.caption(
        f"📄 **CS Resume Status:** {'✅ Loaded' if st.session_state['cs_resume'].strip() else '⚠️ Using Baseline'}\n\n"
        f"📄 **PM Resume Status:** {'✅ Loaded' if st.session_state['pm_resume'].strip() else '⚠️ Using Baseline'}"
    )

# --- Main Header ---
st.title("🎯 Autonomous Career Research & ATS Matcher")
st.markdown(
    "Search direct company career pages & ATS boards on demand, filter out senior/niche roles, "
    "and generate a **90%–95% ATS optimization roadmap** for every listing."
)

tab_batch, tab_single, tab_resumes = st.tabs([
    "🚀 On-Demand Batch Search (15 Roles)",
    "🔍 Single Job URL Deep-Dive",
    "📄 Manage Reference Resumes (CS & PM)",
])

# =====================================================================
# TAB 1: ON-DEMAND BATCH SEARCH & ATS AUDIT
# =====================================================================
with tab_batch:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Run Live Search & ATS Audit", type="primary", use_container_width=True)
    with col_info:
        st.info(
            "Searches Greenhouse, Lever, Ashby, and direct career pages for roles posted in the last "
            f"**{days_posted} days** paying **${min_salary:,}+**, excluding Senior titles and 5+ yr mandates."
        )

    if run_btn:
        if not api_key_input:
            st.error("Please enter your Gemini API Key in the sidebar first.")
        else:
            with st.spinner(f"Searching live ATS boards for {num_jobs} matching roles and running deep ATS audits..."):
                try:
                    results = run_search_and_audit_pipeline(
                        api_key=api_key_input,
                        model_name=model_choice,
                        cs_resume=st.session_state["cs_resume"],
                        pm_resume=st.session_state["pm_resume"],
                        num_jobs=num_jobs,
                        min_salary=min_salary,
                        days_posted=days_posted,
                        roles_list=selected_roles,
                        locations_list=selected_locations,
                        exclusions_text=exclusions_text,
                    )
                    st.session_state["last_batch_results"] = results
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")

    if "last_batch_results" in st.session_state and st.session_state["last_batch_results"].jobs:
        jobs_list: List[JobAuditItem] = st.session_state["last_batch_results"].jobs

        st.subheader(f"📊 Summary Table ({len(jobs_list)} Roles Found)")
        df_rows = []
        for j in jobs_list:
            df_rows.append({
                "Company Name": j.company_name,
                "Job Title": f"{j.job_title} ({j.location_work_style} | {j.compensation_range})",
                "Direct Application Link": j.direct_application_url,
                "Resume Track": j.selected_resume_track,
                "Initial ATS Score": f"{j.initial_ats_score}%",
                "Target Score": f"{j.projected_ats_score}%",
                "ATS Audit Summary (Path to 90%-95%)": j.one_sentence_summary,
            })
        summary_df = pd.DataFrame(df_rows)

        st.dataframe(
            summary_df,
            column_config={
                "Direct Application Link": st.column_config.LinkColumn("Direct Application Link")
            },
            use_container_width=True,
            hide_index=True,
        )

        # CSV Download for Google Sheets
        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV (for Google Sheets Import)",
            data=csv_bytes,
            file_name="career_agent_ats_audit.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("🔬 Detailed ATS Diagnostic Cards & Ready-to-Paste Resume Edits")

        for idx, job in enumerate(jobs_list, start=1):
            with st.expander(
                f"{idx}. {job.company_name} — {job.job_title} | "
                f"Initial ATS: {job.initial_ats_score}% ➔ {job.projected_ats_score}% ({job.selected_resume_track})"
            ):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Initial ATS Score", f"{job.initial_ats_score}%")
                m2.metric("Projected Score", f"{job.projected_ats_score}%", f"+{job.projected_ats_score - job.initial_ats_score}%")
                m3.metric("Compensation", job.compensation_range)
                m4.metric("Location", job.location_work_style)

                st.markdown(f"**🔗 Direct Application Link:** [{job.direct_application_url}]({job.direct_application_url})")
                st.markdown(f"**🏢 Company Stage Verification:** {job.company_stage_note}")

                st.markdown("#### 📐 Weighted ATS Rubric Breakdown")
                r1, r2, r3, r4 = st.columns(4)
                r1.progress(job.rubric_breakdown.hard_skills_score / 100, text=f"Hard Skills (40%): {job.rubric_breakdown.hard_skills_score}%")
                r2.progress(job.rubric_breakdown.responsibilities_score / 100, text=f"Responsibilities (35%): {job.rubric_breakdown.responsibilities_score}%")
                r3.progress(job.rubric_breakdown.experience_scope_score / 100, text=f"Exp & Scope (15%): {job.rubric_breakdown.experience_scope_score}%")
                r4.progress(job.rubric_breakdown.metrics_impact_score / 100, text=f"Metrics (10%): {job.rubric_breakdown.metrics_impact_score}%")

                k_col1, k_col2 = st.columns(2)
                with k_col1:
                    st.markdown("✅ **Matched Keywords Already in Your Resume**")
                    st.write(", ".join(f"`{k}`" for k in job.matched_keywords) if job.matched_keywords else "None identified")
                with k_col2:
                    st.markdown("🚨 **Missing High-Frequency Keywords to Add**")
                    st.write(", ".join(f"`{k}`" for k in job.missing_critical_keywords) if job.missing_critical_keywords else "None")

                st.markdown("#### ✍️ Ready-to-Paste Resume Bullet Rewrites (To Reach 90%–95%)")
                for b in job.tailored_bullet_rewrites:
                    st.markdown(f"**Target Section:** *{b.target_section_or_role}*")
                    st.code(b.rewritten_bullet, language="markdown")
                    st.caption(f"Keywords injected: {', '.join(b.keywords_added)}")

# =====================================================================
# TAB 2: SINGLE JOB URL / JD DEEP-DIVE AUDITOR
# =====================================================================
with tab_single:
    st.subheader("🔍 Audit Any Specific Job URL or Pasted Description")
    single_url = st.text_input("Direct Job Posting URL (Greenhouse, Lever, Ashby, Company Page)")
    single_jd_text = st.text_area("Or Paste Full Job Description Text (Optional if URL is public)", height=180)

    if st.button("🧪 Run Deep-Dive ATS Audit on This Role", type="primary"):
        if not api_key_input:
            st.error("Please enter your Gemini API Key in the sidebar.")
        elif not single_url and not single_jd_text.strip():
            st.warning("Please provide either a job URL or paste the job description text.")
        else:
            with st.spinner("Scraping job description and computing ATS breakdown..."):
                try:
                    single_audit = run_single_job_audit(
                        api_key=api_key_input,
                        model_name=model_choice,
                        cs_resume=st.session_state["cs_resume"],
                        pm_resume=st.session_state["pm_resume"],
                        job_url=single_url,
                        job_text_override=single_jd_text,
                    )
                    st.success(
                        f"**{single_audit.company_name} — {single_audit.job_title}** | "
                        f"Best Resume: **{single_audit.selected_resume_track}** | "
                        f"Initial Score: **{single_audit.initial_ats_score}%** ➔ Target: **{single_audit.projected_ats_score}%**"
                    )
                    st.markdown(f"**Executive Summary:** {single_audit.one_sentence_summary}")
                    st.markdown("**Missing Critical Keywords:** " + ", ".join(f"`{k}`" for k in single_audit.missing_critical_keywords))
                    st.markdown("#### ✍️ Tailored Bullet Rewrites to Hit 90%–95%")
                    for b in single_audit.tailored_bullet_rewrites:
                        st.markdown(f"**{b.target_section_or_role}**")
                        st.code(b.rewritten_bullet, language="markdown")
                except Exception as e:
                    st.error(f"Failed to audit job: {e}")

# =====================================================================
# TAB 3: MANAGE REFERENCE RESUMES (CS & PM)
# =====================================================================
with tab_resumes:
    st.subheader("📄 Upload or Paste Your Reference Resumes")
    st.markdown(
        "Upload your **Customer Success** and **Project Management** resumes here. "
        "Click **Save Resumes Locally** so they remain permanently loaded every time you launch the app."
    )

    col_cs, col_pm = st.columns(2)
    with col_cs:
        st.markdown("### 🤝 Customer Success (CS) Resume")
        cs_file = st.file_uploader("Upload CS Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], key="cs_up")
        if cs_file is not None:
            st.session_state["cs_resume"] = extract_text_from_upload(cs_file)
        cs_text_input = st.text_area("CS Resume Text", value=st.session_state["cs_resume"], height=320)

    with col_pm:
        st.markdown("### 📋 Project Management (PM) Resume")
        pm_file = st.file_uploader("Upload PM Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"], key="pm_up")
        if pm_file is not None:
            st.session_state["pm_resume"] = extract_text_from_upload(pm_file)
        pm_text_input = st.text_area("PM Resume Text", value=st.session_state["pm_resume"], height=320)

    if st.button("💾 Save Resumes Locally", type="primary"):
        st.session_state["cs_resume"] = cs_text_input
        st.session_state["pm_resume"] = pm_text_input
        save_resumes_to_disk(cs_text_input, pm_text_input)
        st.success("Both resumes have been saved to `saved_resumes.json` and will auto-load on every run!")
