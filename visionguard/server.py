    st.caption("The default catalogue is products.csv in the project folder.")

tab_analyze, tab_metrics, tab_about = st.tabs(
    ["Image analysis", "Model performance", "About"]
)

with tab_analyze:
    if uploaded_image is None:
        st.info("Upload a PNG, JPG, JPEG or WEBP image using the sidebar to begin.")
    else:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            left, right = st.columns([1, 1])
            with left:
                st.subheader("Uploaded image")
                st.image(image, use_container_width=True)
            with right:
                st.subheader("Analysis")
                if st.button("Analyze image", type="primary", use_container_width=True):
                    if not MODEL_PATH.exists():
                        st.error(
                            f"Model file not found: {MODEL_PATH}. "
                            "Make sure models/image_detector.keras is present."
                        )
                    else:
                        try:
                            catalogue = load_products(uploaded_csv)
                            model, reader = load_engines(str(MODEL_PATH))
                            with st.spinner("Classifying image and reading text..."):
                                result = analyze_image(image, model, reader, catalogue)
                            st.session_state["visionguard_result"] = result
                            st.session_state["visionguard_image"] = image
                        except Exception as exc:
                            st.error(f"Analysis failed: {exc}")

            result = st.session_state.get("visionguard_result")
            if result is not None:
                st.divider()
                p_real = result["p_real"]
                is_real = p_real >= 0.5
                confidence = max(p_real, 1 - p_real)
                a, b, c = st.columns(3)
                a.metric("Prediction", "Camera-captured / REAL" if is_real else "AI-generated / FAKE")
                b.metric("Confidence", f"{confidence * 100:.1f}%")
                c.metric("Estimated REAL probability", f"{p_real * 100:.1f}%")
                st.progress(confidence, text="Confidence in the predicted class")

                if show_boxes and result["detections"]:
                    st.subheader("Detected text regions")
                    st.image(draw_ocr_boxes(image, result["detections"]), use_container_width=True)

                st.subheader("Text found (OCR)")
                if result["detections"]:
                    ocr_frame = pd.DataFrame([
                        {"Text": d["text"], "OCR confidence": f'{d["confidence"]:.1f}%'}
                        for d in result["detections"]
                    ])
                    st.dataframe(ocr_frame, use_container_width=True, hide_index=True)
                    st.text_area(
                        "Extracted text",
                        value="\n".join(d["text"] for d in result["detections"]),
                        height=120,
                    )
                else:
                    st.info("No readable text was detected.")

                st.subheader("Closest product matches")
                if result["matches"]:
                    st.dataframe(
                        pd.DataFrame(result["matches"]).head(10),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No product matches were found because no readable text was detected.")
                st.caption(f'Analysis completed in {result["elapsed"]:.2f} seconds.')
                st.caption("Product similarity compares printed text only; it does not verify genuineness.")
        except Exception as exc:
            st.error(f"Could not open the uploaded image: {exc}")

with tab_metrics:
    show_evaluation()

with tab_about:
    st.header("How VisionGuard works")
    st.markdown(
        """
        1. **Classify:** the saved TensorFlow model estimates whether the image resembles
           the REAL class or the AI-generated class in its training data.
        2. **Read:** EasyOCR extracts visible text and its location in the image.
        3. **Match:** RapidFuzz compares extracted text with product name, brand and
           net quantity in the reference CSV.

        **Important:** CIFAKE is a general real-versus-synthetic image dataset, not a
        product-authenticity dataset. Lighting, compression, camera type and unseen
        generators can affect predictions. Validate with representative product images
        before using this system in a real-world workflow.
        """
    )
