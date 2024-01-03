use std::process::Command;
use actix_web::{HttpResponse, web};
use serde::{Deserialize, Serialize};
use crate::{DATA_PATH, INTERPRET};

#[derive(Serialize, Deserialize)]
pub struct HandleRequest {
    file_name: String,
    row_index: String,
    col_index: String,
    col_names: String,
    method: String,
    nil: String,
    save_name: String,
}

pub async fn handle_req(info: web::Json<HandleRequest>) -> actix_web::Result<HttpResponse> {
    let command = Command::new(INTERPRET)
        .args(&[
            "./lib/libs/run.py",
            "--data_deal",
            "1",
            "--delete_row_by_index",
            &info.row_index,
            "--delete_column_by_index",
            &info.col_index,
            "--delete_column_by_names",
            &info.col_names,
            "--data_standard",
            &info.method,
            "--data_nil",
            &info.nil,
            "--data_path",
            &format!("{}/{}", DATA_PATH, info.file_name),
            "--data_out_path",
            &format!("{}/{}", DATA_PATH, info.save_name),
        ])
        .output();

    println!("{:#?}", command.unwrap());

    Ok(
        HttpResponse::Ok()
            .body("Ok")
    )
}