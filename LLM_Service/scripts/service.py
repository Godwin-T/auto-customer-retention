import os
from constant import MODEL_NAME, API_KEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def get_context():

    context = "Generate a mail for one of our customers in a relationship informing them of a network upgrade coming to the area soon"
    return context


def mail_generation(model, context):

    prompt = PromptTemplate(
        input_variables=["description"],
        template="We are a telecommunication company and we are trying to prevent customers from churning\
                                . Generate a promotional mail for one of our customer using information in {description} as \
                                    context and it should have a subject relating to the context",
    )
    output = model.invoke(prompt.format(description=context))
    return output


def mail_revamp(model, mail, corrections):

    prompt = PromptTemplate(
        input_variables=["mail" "corrections"],
        template="The {mail} you generated is fine but can these {corrrections} be made to make it better",
    )
    output = model.invoke(prompt.format(mail=mail, corrections=corrections))
    return output


def main():

    os.environ[[REDACTED]] = API_KEY
    model = OpenAI(model=MODEL_NAME, temperature=0.5)
    context = get_context()
    mail = mail_generation(model, context)
    print(mail)


if __name__ == "__main__":
    main()
