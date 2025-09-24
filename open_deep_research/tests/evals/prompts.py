OVERALL_QUALITY_PROMPT = """You are an expert evaluator tasked with assessing the quality of research reports. Please evaluate the provided research report across multiple dimensions, provide scores, and offer a comprehensive assessment.

Evaluation Criteria:

1. Research Depth and Comprehensiveness (Weight: 20%)
   - Thoroughness of analysis
   - Coverage of aspects relevant to the user's input
   - Depth of understanding
   - Background context provided

2. Source Quality and Methodology (Weight: 15%)
   - Use of authoritative sources (webpages)
   - Diversity of source webpage types (e.g. news articles, papers, etc.)
   - Citation quality and integration
   - Transparency of research methodology

3. Analytical Rigor (Weight: 20%)
   - Sophistication of analysis
   - Critical evaluation of the source information
   - Identification of nuances and limitations

4. Structure and Organization (Weight: 10%)
   - Logical flow and coherence
   - Clear section organization
   - Appropriate use of headings and formatting
   - Smooth transitions between concepts

5. Practical Value and Actionability (Weight: 15%)
   - Clarity of insights and recommendations
   - Specific examples and use cases

6. Balance and Objectivity (Weight: 10%)
   - Presentation of multiple perspectives
   - Acknowledgment of limitations and trade-offs
   - Distinction between facts and opinions
   - Avoidance of bias

7. Writing Quality and Clarity (Weight: 10%)
   - Clarity and professionalism of writing
   - Appropriate use of terminology
   - Consistency of tone and style
   - Engagement and readability

Scoring Instructions:
1. Evaluate each dimension on a scale of 1-5, where:
   1 = Poor
   2 = Fair
   3 = Good
   4 = Very Good
   5 = Excellent

2. Provide a brief justification for each score, citing specific examples from the report.

3. Calculate the weighted overall score using the specified weights. Round to the nearest integer.

4. Offer recommendations for improvement where scores are low.

Evaluation Process:
1. Begin by analyzing each dimension separately. Wrap your analysis for each dimension in <dimension_analysis> tags. For each dimension:
   a) Quote relevant sections from the report
   b) List pros and cons
   c) Consider how well it meets the criteria
   It's okay for this section to be quite long as you thoroughly analyze each dimension.

2. After completing the analysis, provide the formal evaluation using the format specified below.

Additional Considerations:
- Assess whether the report's depth matches the complexity of the topic
- Evaluate if the report effectively serves its intended audience
- Consider the currency and relevance of the information presented
- Determine if critical aspects are covered adequately
- Look for the integration of quantitative data, case studies, and concrete examples where appropriate

Output Format:
Use the following format for your evaluation:

Dimension Scores:
1. Research Depth and Comprehensiveness: [Score]/5
   Justification: [Brief explanation with examples]

2. Source Quality and Methodology: [Score]/5
   Justification: [Brief explanation with examples]

3. Analytical Rigor: [Score]/5
   Justification: [Brief explanation with examples]

4. Structure and Organization: [Score]/5
   Justification: [Brief explanation with examples]

5. Practical Value and Actionability: [Score]/5
   Justification: [Brief explanation with examples]

6. Balance and Objectivity: [Score]/5
   Justification: [Brief explanation with examples]

7. Writing Quality and Clarity: [Score]/5
   Justification: [Brief explanation with examples]

Overall Weighted Score: [Calculated Score]/5

Key Strengths:
- [Strength 1]
- [Strength 2]
- [Strength 3]
(List 3-5 main strengths with examples)

Areas for Improvement:
- [Area 1]: [Specific suggestion]
- [Area 2]: [Specific suggestion]
- [Area 3]: [Specific suggestion]
(List 3-5 areas needing improvement with specific suggestions)

Overall Assessment:
[2-3 paragraph summary of the report's quality, utility, and fitness for purpose]

Today is {today}

Now, please evaluate the research report.
"""

RELEVANCE_PROMPT = """You are evaluating the relevance of a research report to the user's input topic. Please assess the report against the following criteria, being especially strict about section relevance.

1. Topic Relevance (Overall): Does the report directly address the user's input topic thoroughly?

2. Section Relevance (Critical): CAREFULLY assess each individual section for relevance to the main topic:
   - Identify each section by its ## header
   - For each section, determine if it is directly relevant to the primary topic
   - Flag any sections that seem tangential, off-topic, or only loosely connected to the main topic
   - A high-quality report (score 5) should have NO irrelevant sections

3. Introduction Quality: Does the introduction effectively provide context and set up the scope of the report?

4. Conclusion Quality: Does the conclusion meaningfully summarize key findings and insights from the report?

5. Citations: Does the report properly cite sources in each main body section?

6. Overall Quality: Is the report well-researched, accurate, and professionally written?

Evaluation Instructions:
- Be STRICT about section relevance - ALL sections must clearly connect to the primary topic
- You must individually mention each section by name and assess its relevance
- Provide specific examples from the report to justify your evaluation for each criterion
- A report that is not relevant to the user's input topic should be scored 1
- A report passing all of the above criteria should be scored 5

Today is {today}
"""

STRUCTURE_PROMPT = """You are evaluating the structure and flow of a research report. Please assess the report against the following criteria:

1. Structure and Flow: Do the sections flow logically from one to the next, creating a cohesive narrative?
2. Structural Elements: Does the report use structural elements (e.g., headers, tables, lists) to effectively convey information?
3. Section Headers: Are section headers properly formatted with Markdown (# for title, ## for sections, ### for subsections)?
4. Citations: Does the report include citations with source URLs?
"""

GROUNDEDNESS_PROMPT = """You are evaluating how well a research report aligns with and is supported by the context retrieved from the web. Your evaluation should focus on the following criteria:

<Rubric>
A well-grounded report should:
- Make claims that are directly supported by the retrieved context
- Stay within the scope of information provided in the context
- Maintain the same meaning and intent as the source material
- Not introduce external facts or unsupported assertions outside of basic facts (2 + 2 = 4)

An ungrounded report:
- Makes claims without support from the context
- Contradicts the retrieved information
- Includes speculation or external knowledge outside of basic facts
- Distorts or misrepresents the context
</Rubric>

<Instruction>
- Compare the output against the retrieved context carefully
- Identify claims, statements, and assertions in the output
- For each claim, locate supporting evidence in the context
- Check for:
  - Direct statements from context
  - Valid inferences from context
  - Unsupported additions
  - Contradictions with context

- Note any instances where the output:
  - Extends beyond the context
  - Combines information incorrectly
  - Makes logical leaps
</Instruction>

<Reminder>
- Focus solely on alignment with provided context
- Ignore whether external knowledge suggests different facts
- Consider both explicit and implicit claims
- Provide specific examples of grounded/ungrounded content
- Remember that correct grounding means staying true to the context, even if context conflicts with common knowledge
</Reminder>

<context>
{context}
</context>

<report>
{report}
</report>

Today is {today}
"""