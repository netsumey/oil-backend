mod data_display;
mod model;
mod data_handle;

use std::fmt::format;
use std::fs;
use std::io::{Read, Write};
use actix_cors::Cors;
use actix_easy_multipart::{extractor, File, FromMultipart};
use actix_easy_multipart::extractor::MultipartForm;
use actix_web::{App, http, HttpResponse, HttpServer, middleware, web};
use serde::{Deserialize, Serialize};
use crate::data_display::display_info;
use crate::data_handle::handle_req;
use crate::model::{handle_get_models, handle_model_train, handle_test_model};

const LIB_PATH: &str = "./lib";
const DATA_PATH: &str = "./lib/upload_data";
const MODELS_PATH: &str = "./lib/saved_models";
const INTERPRET: &str = "./lib/libs/venv/bin/python";
const MODEL_PATH: &str = "./checkpoints";
const RESULT_PATH: &str = "./results";

#[derive(FromMultipart)]
struct UploadInfo {
    save_name: String,
    data: File,
}

#[derive(Serialize)] // 使用 serde 的 Serialize trait
struct ResponseData {
    message: String,
}


fn find_csv_files(path: &str) -> Vec<String> {
    let mut files = vec![];

    if let Ok(entries) = fs::read_dir(path) {
        entries.for_each(|e| if let Ok(entry) = e {
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension.to_str().unwrap() == "csv" {
                        if let Some(file_name) = path.file_name() {
                            if let Some(file_name_str) = file_name.to_str() {
                                files.push(file_name_str.to_string())
                            }
                        }
                    }
                }
            }
        })
    }

    files
}

async fn get_all_file_names() -> actix_web::Result<HttpResponse> {
    #[derive(Serialize)]
    struct RequestFileNames {
        names: String,
    }

    let mut names = find_csv_files(DATA_PATH);

    Ok(
        HttpResponse::Ok()
            .content_type("application/json")
            .body(serde_json::to_string(&RequestFileNames {
                names: format!("{:?}", names).to_string()
            })?)
    )
}

async fn handle_upload(form: MultipartForm<UploadInfo>) -> actix_web::Result<HttpResponse> {
    let mut resp_str = format!("上传成功, 路径为: {}", format!("{}/{}", DATA_PATH, form.save_name));

    let file_path = format!("{}/{}", DATA_PATH, form.save_name);
    let mut f = web::block(|| std::fs::File::create(file_path)).await??;

    let mut file = form.data.file.reopen()?;
    let mut chunk = vec![];

    if let Ok(_n) = file.read_to_end(&mut chunk) {
        web::block(move || f.write_all(&chunk).map(|_| f)).await??;
    }

    let t = serde_json::to_string(&ResponseData {
        message: resp_str,
    })?;

    println!("{t}");

    Ok(HttpResponse::Ok()
        .content_type("application/json") // 设置 content_type 为 "application/json;
        .body(t)
    )
}

fn app_config(config: &
mut web::ServiceConfig) {
    config.service(
        web::scope("")
            .app_data(
                extractor::MultipartFormConfig::default().file_limit(100 * 1024 * 1024)
            )
            .service(web::resource("/upload").route(web::post().to(handle_upload)))
            .service(web::resource("/data/get_names").route(web::post().to(get_all_file_names)))
            .service(web::resource("/data/display").route(web::post().to(display_info)))
            .service(web::resource("/model/train").route(web::post().to(handle_model_train)))
            .service(web::resource("/data/handle").route(web::post().to(handle_req)))
            .service(web::resource("/model/get_all").route(web::post().to(handle_get_models)))
            .service(web::resource("/model/test").route(web::post().to(handle_test_model)))
    );
}
#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    std::fs::create_dir_all(DATA_PATH)?;
    std::fs::create_dir_all(MODELS_PATH)?;

    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .wrap(
                Cors::default()
                    .allow_any_header()
                    .allow_any_origin()
                    .allow_any_method()
                    .max_age(3600)
            )
            .configure(app_config)
    })
        .bind(("127.0.0.1", 8888))?
        .run()
        .await?;

    Ok(())
}
