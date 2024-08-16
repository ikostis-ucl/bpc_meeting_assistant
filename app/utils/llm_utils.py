from llama_index.core import PromptTemplate


def format_query(query_str):
    query_gen_prompt_str = (
        "Vous êtes un(e) assistant(e) qui aide un chef de projet à extraire des informations de documents qui incluent "
        "le déroulement de réunions autour d'un certain projet. Il s'agit de la requête du chef de projet:\n"
        "Requête: {query}\n"
        "Répondez de manière aussi cohérente que possible. Votre réponse doit être rédigée en français."
    )

    query_gen_prompt = PromptTemplate(query_gen_prompt_str)

    fmt_prompt = query_gen_prompt.format(query=query_str)

    return fmt_prompt
