
CREATE TABLE tracking_orders (
	track_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
	order_id uuid NOT NULL,
	original_order_id VARCHAR(63),
	carrier_name VARCHAR(63) NOT NULL,
	current_order_status VARCHAR(31),
	current_order_status_code VARCHAR(31),
	tracking_link VARCHAR(255),
	status_dates JSONB,
	awbno VARCHAR(63),
	origin_city VARCHAR(63),
	destination_city VARCHAR(63),
	ordered_date DATE,
	ordered_time VARCHAR(15),
	expected_delivery_date VARCHAR(31),
	last_updated_time VARCHAR(15),
	last_updated_date DATE,
	shipping_zone VARCHAR(15),
	feedback_link VARCHAR(255),
	carrier_nickname VARCHAR(51),
	sla_breach BOOLEAN NOT NULL DEFAULT FALSE,
	pickup_date VARCHAR(15),
	ofd_date VARCHAR(15)
);

CREATE TABLE orders (
	order_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
	user_id uuid NOT NULL,
	carrier VARCHAR(31) NOT NULL,
	carrier_id VARCHAR(31) NOT NULL,
	valid_key uuid DEFAULT uuid_generate_v4(),
	account_type VARCHAR(31)
);

CREATE TABLE order_details (
	details_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
	order_id uuid NOT NULL,
	awb_no VARCHAR(63),
	original_order_id VARCHAR(63),
	order_items JSONB,
	carrier VARCHAR(31),
	order_created_date DATE,
	order_created_time VARCHAR(15),
	warehouse_name VARCHAR(127),
	customer_name VARCHAR(255),
	customer_address VARCHAR(300),
	customer_phone VARCHAR(31),
	customer_alternate_phone VARCHAR(31),
	customer_pincode VARCHAR(15),
	customer_city VARCHAR(63),
	customer_state VARCHAR(63),
	customer_email VARCHAR(63),
	dimensions JSONB,
	invoice_value VARCHAR(31),
	invoice_number VARCHAR(63),
	cod_amount VARCHAR(15),
	reference_number VARCHAR(63),
	taxable_value VARCHAR(15),
	tax_percentage VARCHAR(15),
	order_type VARCHAR(15),
	order_mode VARCHAR(15),
	ewaybill_serial_number VARCHAR(31),
	hsn_code VARCHAR(31),
	gst_details JSONB,
	price_of_shipment VARCHAR(31),
	marketplace VARCHAR(63),
	order_note VARCHAR(255),
	tags VARCHAR(255),
	is_hyperlocal_eligible BOOLEAN DEFAULT FALSE,
	customer_lat VARCHAR(31),
	customer_long VARCHAR(31),
	customer_country VARCHAR(31)
);


-- The below tables are less relevant

CREATE TABLE pickup_locations (
	warehouse_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
	user_id uuid NOT NULL,
	warehouse_name VARCHAR(255),
	warehouse_manager_name VARCHAR(255),
	email VARCHAR(255),
	phone VARCHAR(31),
	address_line_1 VARCHAR(1023),
	address_line_2 VARCHAR(1023),
	city_name VARCHAR(255),
	state_name VARCHAR(255),
	country_name VARCHAR(255),
	pincode VARCHAR(31),
	lat VARCHAR(63),
	long VARCHAR(63),
	special_id VARCHAR(255),
	is_return BOOLEAN DEFAULT FALSE,
	is_deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE ndr_orders (
	ndr_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
	track_id uuid NOT NULL,
	ndr_history JSONB,
	current_ndr_status VARCHAR(31),
	current_ndr_attempt VARCHAR(15),
	current_ndr_reason VARCHAR(255),
	action_status VARCHAR(31)
);