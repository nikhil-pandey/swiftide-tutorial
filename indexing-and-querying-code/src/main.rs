use swiftide::traits::SimplePrompt;
use std::{path::PathBuf, str::FromStr};
use dotenv::dotenv;

use anyhow::{Context as _, Result};
use clap::Parser;
use indoc::formatdoc;
use qdrant_client::qdrant::{ReadConsistency, SearchBatchPoints, SearchPointsBuilder};
use swiftide::{
    indexing::Pipeline,
    integrations::{openai::OpenAI, qdrant::Qdrant, redis::Redis, treesitter::SupportedLanguages},
};
use swiftide::indexing::EmbeddingModel;
use swiftide::indexing::loaders::FileLoader;
use swiftide::indexing::transformers::{ChunkCode, ChunkMarkdown, Embed, MetadataQACode, MetadataQAText, OutlineCodeTreeSitter};
use tracing::info;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    language: String,

    #[arg(short, long, default_value = "./")]
    path: PathBuf,

    query: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    dotenv().ok();

    let args = Args::parse();

    let openai = OpenAI::builder()
        .default_embed_model("text-embedding-3-small")
        .default_prompt_model("gpt-4o-mini")
        .build()?;

    let qdrant = Qdrant::builder()
        .vector_size(1536)
        .collection_name("swiftide-tutorial")
        .collection_name("swiftide-tutorial")
        .build()?;

    index_all(&args.language, &args.path, &openai, &qdrant).await?;

    let response = query(&openai, &args.query).await?;
    println!("{}", response);

    Ok(())
}

async fn index_all(language: &str, path: &PathBuf, openai: &OpenAI, qdrant: &Qdrant) -> Result<()> {
    info!(path=?path, language, "Indexing code");

    let language = SupportedLanguages::from_str(language)?;
    let mut extensions = language.file_extensions().to_owned();
    extensions.push("md");

    // By default this is what the db schema looks like:
    // {
    // {
    //   "last_updated_at": "2024-10-09T20:45:16.474647+00:00",
    //   "content": "...",
    //   "path": "/path/to/file.rs",
    // }
    let (mut markdown, mut code) =
        Pipeline::from_loader(FileLoader::new(path).with_extensions(&extensions))
            .with_concurrency(50)
            .filter_cached(Redis::try_from_url(
                "redis://localhost:6397",
                "swiftide-tutorial",
            )?)
            .split_by(|node| {
                // Any errors at this point we just pass to 'markdown'
                let Ok(node) = node else { return true };

                // On true we go 'markdown', on false we go 'code'.
                node.path.extension().map_or(true, |ext| ext == "md")
            });

    code = code
        // Uses tree-sitter to extract best effort blocks of code. We still keep the minimum
        // fairly high and double the chunk size
        // SMALLER CHUNKS
        // This adds metadata to the schema that contains high level structure of the file
        // Each node originating from the same file will have the full structure of the file
        .then(
            OutlineCodeTreeSitter::try_for_language(
                language,
                // Will only generate metadata if it exceeds this minimum size
                Some(5),
            )?,
        )
        // FULL CHUNKS
        // This chunks the code into smaller pieces i.e. nodes
        .then_chunk(ChunkCode::try_for_language_and_chunk_size(
            language,
            50..1024,
        )?)
        // This add a new field to the schema
        // {
        //   ...
        //   "Questions and Answers (code)": "``` Q1: What does this code do? A1: It maps file extensions of various programming languages and markup languages to their corresponding language names. Q2: What other internal parts does the code use? A2: The code uses a match statement to evaluate the file extension and return a Result type indicating the language name or an error. Q3: Does this code have any dependencies? A3: The code relies on the enum or struct types Result and ClipboardError, but does not include any external dependencies. Q4: What are some potential use cases for this code? A4: This code can be used in text editors or IDEs to determine the language of a file based on its extension for syntax highlighting or language-specific features. Q5: What file extensions are recognized by this code? A5: The code recognizes several file extensions including ".rs", ".py", ".js", ".html", ".css", and many more for various programming and markup languages. ```",
        // }
        .then(MetadataQACode::new(openai.clone()));

    markdown = markdown
        .then_chunk(ChunkMarkdown::from_chunk_range(50..1024))
        // Generate questions and answers and them to the metadata of the node
        // This add a new field to the schema
        // {
        //   ...
        //   "Questions and Answers (text)": "```\nQ1: What type of license does this project use?\nA1: The project uses the GNU General Public License v3.0.\n\nQ2: Where can the details of the license be found?\nA2: The details of the license can be found in the LICENSE file.\n\nQ3: What does the GNU General Public License v3.0 allow users to do?\nA3: The GNU General Public License v3.0 allows users to freely use, modify, and distribute the software as long as the same license is maintained.\n\nQ4: Is there any specific version mentioned for the license used in this project?\nA4: Yes, the specific version mentioned is v3.0.\n\nQ5: What implications does using the GNU General Public License v3.0 have for derivative works?\nA5: Derivative works must also be licensed under the same GNU General Public License v3.0.\n```",
        // }
        .then(MetadataQAText::new(openai.clone()));

    code.merge(markdown)
        .then_in_batch(Embed::new(openai.clone()).with_concurrency(5).with_batch_size(50))
        .then_store_with(qdrant.clone())
        .run()
        .await
}

async fn query(openai: &OpenAI, question: &str) -> Result<String> {
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_err| "http://localhost:6334".to_string());
    let qdrant_api_key = std::env::var("QDRANT_API_KEY").context("QDRANT_API_KEY not set")?;

    // Build a manual client as Swiftide does not support querying yet
    let qdrant_client = qdrant_client::Qdrant::from_url(&qdrant_url)
        .api_key(qdrant_api_key)
        .build()?;

    // Use Swiftide's openai to rewrite the prompt to a set of questions
    let transformed_question = openai.prompt(formatdoc!(r"
        Your job is to help a code query tool finding the right context.

        Given the following question:
        {question}

        Please think of 5 additional questions that can help answering the original question. The code is written in {lang}.

        Especially consider what might be relevant to answer the question, like dependencies, usage and structure of the code.

        Please respond with the original question and the additional questions only.

        ## Example

        - {question}
        - Additional question 1
        - Additional question 2
        - Additional question 3
        - Additional question 4
        - Additional question 5
        ", question = question, lang = "rust"
    ).into()).await?;

    info!("{}", "=".repeat(80));
    info!("Transformed question: {}", transformed_question);
    info!("{}", "=".repeat(80));

    // Embed the full rewrite for querying
    let embedded_question = openai
        .embed(vec![transformed_question.clone()])
        .await?
        .pop()
        .context("Expected embedding")?;

    info!("{}", "=".repeat(80));
    // This is an array of embedding values
    info!("Embedded question: {:?}", embedded_question);
    info!("{}", "=".repeat(80));

    // Search for matches
    let answer_context_points = qdrant_client
        .search_points(
            SearchPointsBuilder::new("swiftide-tutorial", embedded_question, 20).with_payload(true),
        )
        .await?;

    // we can also do batch search
    // let filter = Filter::must([Condition::matches("city", "London".to_string())]);
    //
    // let searches = vec![
    //     SearchPointsBuilder::new("my_collection", vec![0.2, 0.1, 0.9, 0.7], 3)
    //         .filter(filter.clone())
    //         .build(),
    //     SearchPointsBuilder::new("my_collection", vec![0.5, 0.3, 0.2, 0.3], 3)
    //         .filter(filter)
    //         .build(),
    // ];
    //
    // qdrant_client
    //     .search_batch_points(SearchBatchPointsBuilder::new("my_collection", searches))
    //     .await?;

    info!("{}", "=".repeat(80));
    // SearchResponse {
    //     result: [
    //         ScoredPoint {
    //             id: Some(PointId { point_id_options: Some(Uuid("UUID_HERE")) }),
    //             payload: {
    //                 "last_updated_at": Value { kind: Some(StringValue("TIMESTAMP_HERE")) },
    //                 "path": Value { kind: Some(StringValue("FILE_PATH_HERE")) },
    //                 "content": Value { kind: Some(StringValue("SHORT_CONTENT_HERE")) },
    //                 "Questions and Answers (text)": Value { kind: Some(StringValue("Q&A_TEXT_HERE")) }
    //             },
    //             score: SCORE_HERE,
    //             version: VERSION_HERE,
    //             vectors: None,
    //             shard_key: None,
    //             order_value: None
    //         },
    //         ScoredPoint {
    //             id: Some(PointId { point_id_options: Some(Uuid("UUID_HERE")) }),
    //             payload: {
    //                 "last_updated_at": Value { kind: Some(StringValue("TIMESTAMP_HERE")) },
    //                 "path": Value { kind: Some(StringValue("FILE_PATH_HERE")) },
    //                 "content": Value { kind: Some(StringValue("SHORT_CONTENT_HERE")) },
    //                 "Questions and Answers (code)": Value { kind: Some(StringValue("Q&A_CODE_HERE")) }
    //             },
    //             score: SCORE_HERE,
    //             version: VERSION_HERE,
    //             vectors: None,
    //             shard_key: None,
    //             order_value: None
    //         }
    //         // ... More ScoredPoint entries follow
    //     ],
    //     time: TIME_HERE
    // }
    info!("Answer context points: {:?}", answer_context_points);
    info!("{}", "=".repeat(80));

    // Concatenate all the found chunks
    let answer_context = answer_context_points
        .result
        .into_iter()
        .map(|v| v.payload.get("content").unwrap().to_string())
        .collect::<Vec<_>>()
        .join("\n\n");

    info!("{}", "=".repeat(80));
    info!("Answer context: {}", answer_context);
    info!("{}", "=".repeat(80));

    // A prompt for answering the initial question with the found context
    let prompt = formatdoc!(
        r#"
        Answer the following question(s):
        {question}

        ## Constraints
        * Only answer based on the provided context below
        * Always reference files by the full path if it is relevant to the question
        * Answer the question fully and remember to be concise
        * Only answer based on the given context. If you cannot answer the question based on the
            context, say so.
        * Do not make up anything, especially code, that is not included in the provided context

        ## Context:
        {answer_context}
        "#,
    );

    info!("{}", "=".repeat(80));
    info!("Prompt: {}", prompt);
    info!("{}", "=".repeat(80));

    let answer = openai.prompt(prompt.into()).await?;
    info!("{}", "=".repeat(80));
    info!("{}", answer);
    info!("{}", "=".repeat(80));

    Ok(answer)
}
