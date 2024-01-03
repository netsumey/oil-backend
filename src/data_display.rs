use std::process::{Command, Stdio};
use actix_web::{HttpResponse, web};
use serde::{Deserialize, Serialize};
use crate::{DATA_PATH, INTERPRET};

#[derive(Serialize, Deserialize)]
pub struct DisplayRequest {
    name: String,
    count: bool,
    row_name: bool,
    amount: String,
}

pub async fn display_info(
    info: web::Json<DisplayRequest>
) -> actix_web::Result<HttpResponse> {
    let (name, count, row_name, amount) =
        (info.name.clone(), info.count, info.row_name,
         info.amount.parse::<usize>().unwrap_or(0));

    // println!("1");

    let command = Command::new(INTERPRET)
        .args(&["./lib/libs/run.py", "--get", "1", "--get_amount", &count.to_string(),
        "--get_names", &row_name.to_string(), "--get_top_k", &amount.to_string(),
        "--data_path", &format!("{}/{}", DATA_PATH, name)])
        .output();

    // println!("{:#?}", command);


    Ok(
        HttpResponse::Ok()
            .content_type("text/plain")
            .body(String::from_utf8_lossy(&command.unwrap().stdout).to_string())
    )
}