

-- tracking_orders table contains the data regarding the journey of the order
-- the current_order_status has these possible values: 'CANCELLED', 'ORDERED', 'PICKED UP', 'IN TRANSIT', 'OUT FOR DELIVERY', 'DELIVERED', 'UNDELIVERED', 'RTO IN TRANSIT', 'RTO DELIVERED', 'RTO', 'RTO OFD', 'DELAYED', 'LOST', 'DAMAGED', 'OUT FOR PICKUP', 'PICKUP RESCHEDULED', 'PICKUP EXCEPTION', 'RTO INITIATED', 'RTO NDR', 'MISROUTED', 'CHANNEL_ORDER'

CREATE TABLE tracking_orders (
	track_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
	order_id uuid NOT NULL, -- order identifier
	original_order_id VARCHAR(63), -- seller order id
	carrier_name VARCHAR(63) NOT NULL, -- name of the carrier (courier partner)
	current_order_status VARCHAR(31), -- current tracking status of the shipment
	current_order_status_code VARCHAR(31), -- standardized code for the tracking status
	tracking_link VARCHAR(255), -- link for tracking
	status_dates JSONB, -- this array contains the tracking data of the order journey from hub to hub
	awbno VARCHAR(63), -- awbno is the unique id of the shipment given by the carrier
	origin_city VARCHAR(63),
	destination_city VARCHAR(63),
	ordered_date DATE, -- date when the order is manifested
	ordered_time VARCHAR(15), -- time when the order is manifested
	expected_delivery_date VARCHAR(31),
	last_updated_time VARCHAR(15),
	last_updated_date DATE,
	shipping_zone VARCHAR(15), -- zone of the shipment
	feedback_link VARCHAR(255),
	carrier_nickname VARCHAR(51),
	sla_breach BOOLEAN NOT NULL DEFAULT FALSE, -- if the order is breaching SLA
	pickup_date VARCHAR(15), -- date of pickup
	ofd_date VARCHAR(15) -- date when the shipment is out for delivery
);

-- orders table is the table which connects order_id to the user_id
CREATE TABLE orders (
	order_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY, -- order identifier
	user_id uuid NOT NULL, -- user identifier
	carrier VARCHAR(31) NOT NULL, -- carrier (courier partner)
	carrier_id VARCHAR(31) NOT NULL, -- id of the carrier (courier partner)
	valid_key uuid DEFAULT uuid_generate_v4(),
	account_type VARCHAR(31)
);

-- order_details contains details of the order like customer details, item details, invoice value, cod amount, order type, dimensions, seller order id, etc

CREATE TABLE order_details (
	details_id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
	order_id uuid NOT NULL, -- order identifier
	awb_no VARCHAR(63), -- awb_no is the unique id of the shipment given by the carrier
	original_order_id VARCHAR(63), -- seller order id
	order_items JSONB, -- contains array of order items, it has name, selling price, quantity, sku
	carrier VARCHAR(31), -- name of the carrier (courier partner)
	order_created_date DATE, -- date when the order is manifested
	order_created_time VARCHAR(15), -- time when the order is manifested
	warehouse_name VARCHAR(127), -- name of pickup location
	customer_name VARCHAR(255),
	customer_address VARCHAR(300),
	customer_phone VARCHAR(31),
	customer_alternate_phone VARCHAR(31),
	customer_pincode VARCHAR(15),
	customer_city VARCHAR(63),
	customer_state VARCHAR(63),
	customer_email VARCHAR(63),
	dimensions JSONB, -- json containing length, breadth, height and weight
	invoice_value VARCHAR(31), -- total value of order items
	invoice_number VARCHAR(63),
	cod_amount VARCHAR(15), -- cod amount to be taken from buyer
	reference_number VARCHAR(63),
	taxable_value VARCHAR(15),
	tax_percentage VARCHAR(15),
	order_type VARCHAR(15), -- COD or PREPAID
	order_mode VARCHAR(15), -- SURFACE or EXPRESS
	ewaybill_serial_number VARCHAR(31),
	hsn_code VARCHAR(31),
	gst_details JSONB,
	price_of_shipment VARCHAR(31), -- shipping cost of the shipment
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