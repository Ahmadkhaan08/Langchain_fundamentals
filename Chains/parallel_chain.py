from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

template1=PromptTemplate(
    template="Generate a short and simple notes from the following text \n {text}",
    input_variables=['text']
)

template2=PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

template3=PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n  notes-> {notes} and quiz->{quiz}",
    input_variables=['notes','quiz']
)


model1=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
model2=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
        'notes':template1 | model1 | parser,
        'quiz':template2 | model2 | parser,
    }
)

merge_chain= template3 | model1 | parser
final_chain= parallel_chain | merge_chain | parser

text="""# Superintelligence: The Next Evolution of Artificial Intelligence

## Introduction

Superintelligence refers to a form of intelligence that significantly surpasses the cognitive abilities of the most intelligent human beings across a wide range of fields, including scientific research, logical reasoning, problem-solving, creativity, strategic planning, and decision-making. Unlike conventional artificial intelligence, which is generally designed to perform specific tasks or assist with particular activities, superintelligence represents a hypothetical or future stage of AI capable of outperforming humans in almost every intellectual domain.

The concept of superintelligence is one of the most important and debated topics in modern artificial intelligence research. It offers the possibility of solving some of humanity's most complex problems while also raising important questions about safety, control, ethics, and the future relationship between humans and intelligent machines.

## Understanding Superintelligence

Artificial intelligence has evolved from rule-based systems and narrow machine learning models to advanced systems capable of understanding language, generating content, writing software, analyzing data, and assisting with complex reasoning. However, current AI systems still have significant limitations, including unreliable reasoning, limited real-world understanding, susceptibility to errors, and difficulties with autonomous long-term planning.

Superintelligence goes beyond these capabilities. A superintelligent system would theoretically possess intellectual abilities that exceed human performance across nearly all areas of knowledge and reasoning. It might discover scientific principles that humans have not identified, develop innovative technologies, solve complex mathematical problems, and devise strategies for challenges that are currently beyond human capabilities.

Importantly, superintelligence does not necessarily mean that an AI system would have consciousness, emotions, personal desires, or human-like experiences. Intelligence and consciousness are separate concepts, and a system could theoretically demonstrate extraordinary problem-solving abilities without possessing subjective awareness.

## Levels of Artificial Intelligence

Artificial intelligence is commonly discussed in terms of three broad categories:

### 1. Artificial Narrow Intelligence (ANI)

Artificial Narrow Intelligence is designed to perform specific tasks or a limited range of related tasks. Examples include recommendation systems, image recognition, voice assistants, translation systems, and specialized AI tools for programming or data analysis.

Although narrow AI can outperform humans in certain tasks, it does not automatically possess general intelligence across unrelated domains.

### 2. Artificial General Intelligence (AGI)

Artificial General Intelligence refers to a hypothetical AI system capable of learning, understanding, and performing a broad range of intellectual tasks at a level comparable to humans. Such a system would ideally be able to transfer knowledge between domains, adapt to unfamiliar problems, and learn new skills without requiring a completely separate model for every task.

AGI remains an evolving research goal, and there is no universally accepted test for determining whether a system has achieved it.

### 3. Artificial Superintelligence (ASI)

Artificial Superintelligence describes a hypothetical stage at which AI surpasses human intellectual performance across virtually all relevant cognitive domains. This could include scientific discovery, strategic thinking, engineering, creativity, and complex decision-making.

ASI is not simply an AI that responds faster or remembers more information. Its defining characteristic would be a level of intellectual capability that substantially exceeds the best human abilities across a broad range of activities.

## Potential Capabilities of Superintelligence

If developed successfully and responsibly, superintelligence could potentially transform many aspects of human civilization.

**Scientific discovery:** A superintelligent system might identify new physical principles, develop advanced materials, improve energy technologies, and accelerate research into diseases.

**Healthcare and medicine:** It could help researchers understand complex biological systems, discover promising treatments, design personalized therapies, and improve the efficiency of medical research. These outcomes would still require appropriate validation, clinical trials, and human oversight.

**Engineering and technology:** Superintelligence might design more efficient computers, advanced robotics, intelligent infrastructure, and technologies that are difficult for human researchers to develop independently.

**Education and learning:** It could support highly personalized education, adapting explanations and learning strategies to individual needs while helping learners understand complicated subjects.

**Environmental protection:** Advanced AI could help optimize renewable energy systems, improve climate modeling, reduce waste, and develop better strategies for managing natural resources.

**Economic productivity:** Superintelligent systems might automate complex knowledge work, improve logistics, optimize manufacturing, and accelerate the development of new products and services.

**Problem-solving and decision-making:** By analyzing enormous amounts of information and exploring many possible solutions, such systems could assist with complicated global challenges, provided their objectives and recommendations were appropriately evaluated.

These possibilities are theoretical in the context of superintelligence. They should not be interpreted as capabilities that current AI systems already possess.

## How Could Superintelligence Emerge?

There is no established roadmap for creating superintelligence. Researchers nevertheless explore several possible developments that could contribute to increasingly capable AI systems.

One possibility is continued progress in machine learning, allowing models to develop stronger reasoning, planning, and generalization abilities. Another is the integration of AI systems with tools, memory, scientific software, robotics, and specialized agents.

Researchers also investigate automated AI research, in which AI systems help develop, test, and improve other AI systems. If such processes became sufficiently effective, they might accelerate AI development. However, the extent to which this could happen, and whether it would lead to superintelligence, remains uncertain.

Greater computing power, improved algorithms, high-quality training data, and advances in evaluating AI capabilities could also contribute to future progress. None of these developments individually guarantees that superintelligence will emerge.

## Risks and Challenges

Despite its potential benefits, superintelligence could introduce serious risks if its capabilities developed faster than our ability to understand, evaluate, and control its behavior.

**The alignment problem:** AI alignment concerns ensuring that an AI system's objectives and behavior remain consistent with human intentions and values. A highly capable system pursuing an incorrectly specified objective could produce harmful outcomes even if it followed its instructions precisely.

**Loss of control:** If a future AI system could independently plan, use tools, and pursue long-term objectives, maintaining reliable human oversight could become increasingly challenging. Researchers disagree about the likelihood and severity of these scenarios, but control is an important area of study.

**Economic disruption:** Advanced automation could transform employment, business models, and global markets. While it could create new opportunities and increase productivity, it might also displace workers and concentrate wealth or technological power.

**Security threats:** Powerful AI capabilities could potentially be misused for cyberattacks, sophisticated fraud, surveillance, or other harmful activities. Appropriate safeguards and access controls would be essential.

**Concentration of power:** If a small number of organizations or governments controlled highly advanced AI systems, they could gain disproportionate economic, political, or technological influence.

**Ethical uncertainty:** Superintelligence would raise difficult questions about accountability, human autonomy, privacy, fairness, and who should decide how powerful AI systems are developed and used.

## AI Safety and Alignment

AI safety is a critical field focused on reducing the risks associated with increasingly capable AI systems. Its goals include developing reliable evaluation methods, preventing harmful behavior, improving transparency, and ensuring that AI systems remain subject to appropriate oversight.

Important research areas include interpretability, robustness, adversarial testing, monitoring, access control, and methods for aligning AI behavior with human preferences and societal requirements.

For superintelligence, these challenges may become more difficult because a system that greatly exceeds human capabilities could be harder to evaluate using ordinary testing methods. Researchers therefore investigate how to establish trustworthy safeguards before AI systems reach potentially transformative levels of capability.

Responsible development would also require collaboration among researchers, policymakers, businesses, and the public. Technical safety alone may not be sufficient without effective governance, accountability, and international cooperation.

## Superintelligence and Humanity's Future

The long-term impact of superintelligence is uncertain. Optimistic perspectives emphasize its potential to accelerate scientific progress, reduce scarcity, improve health, and help humanity address major challenges. More cautious perspectives emphasize the possibility of unpredictable behavior, social inequality, misuse, and difficulty maintaining meaningful human control.

These outcomes are not predetermined. They may depend on how advanced AI systems are designed, who controls them, how their benefits are distributed, and whether effective safety measures are established.

Rather than assuming that superintelligence will automatically produce either a utopian or catastrophic future, it is more useful to treat it as a possible technological development that deserves careful research, realistic evaluation, and responsible planning.

## Conclusion

Superintelligence represents a hypothetical future in which artificial intelligence surpasses human intellectual capabilities across nearly every major cognitive domain. It could open new possibilities in science, medicine, education, engineering, and economic development, potentially changing the course of human civilization.

At the same time, developing systems of such capability would raise unprecedented technical, ethical, economic, and governance challenges. Ensuring that increasingly advanced AI remains reliable, secure, beneficial, and aligned with human intentions is therefore a central concern for the future of artificial intelligence.

Ultimately, the significance of superintelligence will not be determined solely by how intelligent machines become, but also by how wisely humanity chooses to develop, govern, and use them.
"""

result=final_chain.invoke({'text':text})
print(result)

# chain.get_graph().print_ascii()