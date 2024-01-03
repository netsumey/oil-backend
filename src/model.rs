use std::fs::read;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use actix_web::{HttpResponse, web};
use serde::{Deserialize, Serialize};
use crate::data_display::DisplayRequest;
use crate::{DATA_PATH, INTERPRET};

#[derive(Serialize, Deserialize)]
pub struct ModelRequest {
    train: String,
    test: String,
    valid: String,
    output_name: String,
    structure: String,
    seq_len: String,
    label_len: String,
    pred_len: String,
    e_layers: String,
    d_layers: String,
    factor: String,
    enc_in: String,
    dec_in: String,
    c_out: String,
    itr: String,
}

pub async fn handle_model_train(req: web::Json<ModelRequest>)
-> actix_web::Result<HttpResponse> {
    let mut command = Command::new(INTERPRET)
        .args(&[
            "./lib/libs/run.py",
            "--task_name",
            "long_term_forecast",
            "--is_training",
            "1",
            "--root_path",
            "./lib/upload_data",
            "--data_train_path",
            &format!("{}", req.train),
            "--data_test_path",
            &format!("{}", req.test),
            "--data_vali_path",
            &format!("{}", req.valid),
            "--model_id",
            &req.output_name,
            "--model",
            &req.structure,
            "--target",
            "press",
            "--features",
            "MS",
            "--seq_len",
            &req.seq_len.to_string(),
            "--label_len",
            &req.label_len.to_string(),
            "--pred_len",
            &req.pred_len.to_string(),
            "--e_layers",
            &req.e_layers.to_string(),
            "--d_layers",
            &req.d_layers.to_string(),
            "--factor",
            &req.factor,
            "--enc_in",
            &req.enc_in,
            "--dec_in",
            &req.dec_in,
            "--c_out",
            &req.c_out,
            "--des",
            "Exp",
            "--itr",
            &req.itr,
        ]).stdout(Stdio::piped())
        .spawn()?;

        println!("{:#?}", command);

    let stdout = command.stdout.take().unwrap();
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        println!("{}", line?);
    }

    let status = command.wait()?;

    println!("return");

    Ok(HttpResponse::Ok().body(""))
}